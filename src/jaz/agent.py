import html
import re
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple, overload

from typing_extensions import TypeForm

if TYPE_CHECKING:
    from .invoke import Prehook

from tenacity import (
    RetryCallState,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .budget import CostTracker
from .config import Config, get_config
from .exceptions import (
    BudgetExhaustedError,
    BudgetForcingException,
    LLMOutputParseError,
    _JazInternalError,
)
from .hooks import get_dispatcher
from .hooks.events import (
    BudgetUpdate,
    InvokeEnter,
    LLMQueryEnter,
    LLMRetry,
    Message,
    ReplExecutionEnter,
    ReplIterationEnter,
)
from .library import Library
from .llm_client import LLMClient, LLMResponse, create_llm_client
from .prompts import get_system_prompt, get_user_prompt
from .providers import MessageDict
from .repl import REPL, REPL_LANGUAGE_MAP
from .repl.registry import REPL_LANG_NAME_PAT
from .repl.types import (
    ErrorResult,
    ExecResult,
    ExecuteResult,
    RaiseResult,
    ReturnResult,
)
from .string_utils import abbreviate_string


def _raise_halt_errors(errors: list[Exception]) -> None:
    """Raise halt errors from hooks. Single error raised directly; multiple as ExceptionGroup."""
    if len(errors) == 1:
        raise errors[0]
    raise ExceptionGroup("multiple hooks requested halt", errors)


# Regex to match REPL input with language specification in lang attribute.
# Accept both single and double quotes for the attribute.
# Closing tag is optional to handle truncated LLM responses (matches to end of string).
# The language name character class is shared with REPL_LANG_NAME_PAT in repl/registry.py.
# Groups: 1=lang, 2=timeout (optional float, seconds), 3=body
REPL_INPUT_REGEX = re.compile(
    rf'<repl_input\s+lang=["\']({REPL_LANG_NAME_PAT})["\'](?:\s+timeout=["\'](\d+(?:\.\d+)?)["\'])?\s*>(.*?)(?:</repl_input>|\Z)',
    re.DOTALL,
)


class ReplIterationResult[ReturnT](NamedTuple):
    """Result from a single REPL iteration.

    Attributes:
        exec_result: The execution result from the REPL
        repls: Dictionary of available REPLs
        response_content: The LLM response content (None if empty)
        code: The extracted code from the response (None if not found)
    """

    exec_result: ExecResult[ReturnT]
    repls: dict[str, REPL]
    response_content: str | None
    code: str | None


def _dedupe_libraries_by_root_module(libraries: list[Library]) -> list[Library]:
    """Deduplicate libraries by root module name, keeping the last one.

    This preserves current runtime semantics where later libraries overwrite
    earlier ones in the REPL state, while avoiding duplicate tool descriptions
    in prompts when hooks intentionally replace a library namespace.

    Assumption: each Library is effectively used as a single top-level namespace
    provider. If two libraries expose the same root module, we treat the later
    one as the intended replacement for prompt/rendering purposes.
    """
    seen_root_modules: set[str] = set()
    deduped_reversed: list[Library] = []
    for library in reversed(libraries):
        root_modules = set(getattr(library, "_root_modules", {}).keys())
        if root_modules & seen_root_modules:
            continue
        deduped_reversed.append(library)
        seen_root_modules.update(root_modules)
    deduped_reversed.reverse()
    return deduped_reversed


class Agent:
    def __init__(
        self,
        repls: list[str],
        cost_tracker: CostTracker | None = None,
        prehook: "Prehook | None" = None,
        config: Config | None = None,
    ) -> None:
        if not repls:
            raise ValueError("At least one REPL language must be specified")
        self.config = config if config is not None else get_config()
        # Store just the REPL classes - configs come from Config at runtime
        self.repl_classes: dict[str, type[REPL]] = {}
        for language in repls:
            if language not in REPL_LANGUAGE_MAP:
                raise ValueError(f"Unsupported REPL language: {language}")
            self.repl_classes[language] = REPL_LANGUAGE_MAP[language]
        if isinstance(self.config.llm_client, str):
            self.llm_client: LLMClient = create_llm_client(self.config.llm_client)
        elif isinstance(self.config.llm_client, LLMClient):
            self.llm_client = self.config.llm_client
        else:
            raise TypeError(
                f"llm_client must be a string or LLMClient instance, "
                f"got {type(self.config.llm_client).__name__}"
            )
        self.model: str = self.llm_client.get_model(self.config.model_config)
        self.cost_tracker = cost_tracker
        self.prehook = prehook
        self.dispatcher = get_dispatcher()
        self._last_prompt_tokens: int | None = None

    def query(self, system_prompt: str, user_prompt: str) -> str | None:
        messages: list[MessageDict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for message in messages:
            self.dispatcher.process_message(Message(message=message))
        response_content = self._query_with_messages(messages)
        self.dispatcher.process_message(
            Message(message={"role": "assistant", "content": response_content})
        )
        return response_content

    def _query_with_messages(
        self,
        messages: list[MessageDict],
        iteration: int | None = None,
        cur_recursion_depth: int | None = None,
        invoke_id: str | None = None,
    ) -> str | None:
        # Custom retry callback that uses the dispatcher to process the retry event
        def log_retry_attempt(retry_state: RetryCallState):
            exception = retry_state.outcome.exception() if retry_state.outcome else None
            wait_seconds = (
                retry_state.next_action.sleep
                if retry_state.next_action is not None
                else 0.0
            )
            self.dispatcher.process_llm_retry(
                LLMRetry(
                    invoke_id=invoke_id or "",
                    model=self.model,
                    attempt_number=retry_state.attempt_number,
                    exception=exception
                    if isinstance(exception, Exception)
                    else Exception(str(exception)),
                    wait_seconds=wait_seconds,
                    iteration=iteration,
                    cur_recursion_depth=cur_recursion_depth,
                )
            )

        completion_with_retry = retry(
            reraise=True,
            stop=stop_after_attempt(self.config.max_llm_attempts),
            wait=wait_exponential(
                multiplier=self.config.llm_retry_wait_multiplier,
                min=self.config.llm_retry_wait_min,
                max=self.config.llm_retry_wait_max,
            ),
            before_sleep=log_retry_attempt,
            retry=retry_if_not_exception_type(self.llm_client.non_retryable_exceptions),
        )(self.llm_client.complete)

        # Use dispatcher span for LLM query
        # Generate invoke_id if not provided (for standalone queries)
        if invoke_id is None:
            invoke_id = str(uuid.uuid4())
        enter_event = LLMQueryEnter(
            invoke_id=invoke_id,
            messages=messages,
            model=self.model,
            iteration=iteration,
            cur_recursion_depth=cur_recursion_depth,
        )
        with self.dispatcher.span_llm_query(enter_event) as span:
            # Check if hooks want to halt
            if span.ctx.should_halt:
                _raise_halt_errors(span.ctx.halt_errors)

            # Make LLM call
            start_time = datetime.now()
            llm_response: LLMResponse = completion_with_retry(
                messages=messages,
                **self.config.model_config,
            )
            end_time = datetime.now()

            # Extract usage for context window tracking (regardless of cost tracker)
            if llm_response.prompt_tokens is not None:
                self._last_prompt_tokens = llm_response.prompt_tokens

            # Track cost if tracker is available
            if self.cost_tracker is not None:
                assert llm_response.prompt_tokens is not None, (
                    "LLM usage info is missing"
                )
                assert llm_response.completion_tokens is not None, (
                    "LLM usage info is missing"
                )
                assert llm_response.total_tokens is not None, (
                    "LLM usage info is missing"
                )
                assert llm_response.cost is not None, "LLM cost info is missing"

                self.cost_tracker.add_llm_call(
                    model=self.model,
                    prompt_tokens=llm_response.prompt_tokens,
                    completion_tokens=llm_response.completion_tokens,
                    total_tokens=llm_response.total_tokens,
                    cost=llm_response.cost,
                    iteration=iteration,
                    start_time=start_time,
                    end_time=end_time,
                )

                # Log budget status via dispatcher
                llm_cost_budget_status = self.cost_tracker.get_llm_cost_budget_status(
                    include_root=True
                )
                llm_calls_budget_status = self.cost_tracker.get_llm_calls_budget_status(
                    include_root=True
                )
                self.dispatcher.process_budget_update(
                    BudgetUpdate(
                        budget_status=llm_cost_budget_status,
                        cur_recursion_depth=cur_recursion_depth,
                    )
                )
                self.dispatcher.process_budget_update(
                    BudgetUpdate(
                        budget_status=llm_calls_budget_status,
                        cur_recursion_depth=cur_recursion_depth,
                    )
                )

            # Complete the span with response data
            span.complete(
                response_content=llm_response.content,
                prompt_tokens=llm_response.prompt_tokens,
                completion_tokens=llm_response.completion_tokens,
                cost=llm_response.cost,
                iteration=iteration,
                start_time=start_time,
                end_time=end_time,
            )

            return llm_response.content

    def do_one_repl_iteration(
        self,
        iteration: int,
        max_iterations: int,
        cur_recursion_depth: int,
        messages: list[MessageDict],
        repls: dict[str, REPL],
        error_if_not_finish: Exception | None,
        invoke_id: str,
        budget_forcing_active: bool = False,
        instruction_prompt_additions: list[str] | None = None,
    ) -> ReplIterationResult[Any]:
        # TODO: Fix static types
        # If there are instruction prompt additions from hooks, inject them as a
        # temporary user message for this query only
        # NOTE: Currently we inject instruction prompts as a new user message after the REPL execution output message. This might cause problems depending on the LLM where it sees double user messages and gets confused. A more robust way to do this is to inject the instruction prompt additions directly into the last user message (which is the REPL output message) before the LLM query. However, this requires mutating the messages list which is also used for logging and might cause confusion in logs. For now, we will keep it simple and inject as a new user message, but if we see issues with LLM confusion we can switch to mutating the last user message.
        if instruction_prompt_additions:
            additional_message: MessageDict = {
                "role": "user",
                "content": "\n".join(instruction_prompt_additions),
            }
            query_messages: list[MessageDict] = [
                *messages,
                additional_message,
            ]
            self.dispatcher.process_message(Message(message=additional_message))
        else:
            query_messages = messages

        response_content = self._query_with_messages(
            query_messages,
            iteration=iteration,
            cur_recursion_depth=cur_recursion_depth,
            invoke_id=invoke_id,
        )
        messages.append({"role": "assistant", "content": response_content})
        self.dispatcher.process_message(Message(message=messages[-1]))
        if response_content is None:
            # TODO: Is this the best way to handle this?
            error = LLMOutputParseError("LLM returned empty response")
            return ReplIterationResult(
                exec_result=ErrorResult(
                    output="",
                    error_summary=f"{type(error).__name__}: {str(error)}",
                    exception=error,
                ),
                repls=repls,
                response_content=response_content,
                code=None,
            )
        match = REPL_INPUT_REGEX.search(response_content)
        if match is None:
            error = LLMOutputParseError(
                "REPL input not found in response. "
                "Make sure to put your REPL input in XML tags like this:\n"
                '<repl_input lang="language">\n...\n</repl_input>\n'
                "where `language` is the language of your REPL input "
                "(e.g., `python`, `bash`). "
                "Also make sure to escape any XML tags inside your REPL input."
            )
            return ReplIterationResult(
                exec_result=ErrorResult(
                    output="",
                    error_summary=f"{type(error).__name__}: {str(error)}",
                    exception=error,
                ),
                repls=repls,
                response_content=response_content,
                code=None,
            )

        language = match.group(1)
        exec_timeout_override = (
            float(match.group(2)) if match.group(2) is not None else None
        )
        code = match.group(3).strip("\n\r")

        # TODO: optimize this with finditer
        # TODO: lift this constraint
        if len(REPL_INPUT_REGEX.findall(response_content)) > 1:
            error = LLMOutputParseError(
                "Multiple REPL inputs found in response. "
                "Make sure to write only one XML block <repl_input>...</repl_input>."
            )
            return ReplIterationResult(
                exec_result=ErrorResult(
                    output="",
                    error_summary=f"{type(error).__name__}: {str(error)}",
                    exception=error,
                ),
                repls=repls,
                response_content=response_content,
                code=code,
            )

        # Select the appropriate REPL based on the language
        if language not in repls:
            error = LLMOutputParseError(
                f"Unsupported language '{language}'. "
                f"Available languages: {', '.join(repls.keys())}"
            )
            return ReplIterationResult(
                exec_result=ErrorResult(
                    output="",
                    error_summary=f"{type(error).__name__}: {str(error)}",
                    exception=error,
                ),
                repls=repls,
                response_content=response_content,
                code=code,
            )

        repl = repls[language]

        # Unescape HTML entities in code (e.g., &lt; -> <, &gt; -> >)
        # This handles cases where the LLM outputs escaped XML tags
        code = html.unescape(code)

        # Use dispatcher span for REPL iteration
        enter_event = ReplExecutionEnter(
            invoke_id=invoke_id,
            iteration=iteration,
            max_iterations=max_iterations,
            code=code,
            cur_recursion_depth=cur_recursion_depth,
        )
        with self.dispatcher.span_repl_execution(enter_event) as span:
            # Check if hooks want to halt
            if span.ctx.should_halt:
                _raise_halt_errors(span.ctx.halt_errors)

            # Apply execution-level REPL inputs from hooks
            if span.ctx.repl_inputs:
                repl.add_inputs(span.ctx.repl_inputs)

            # Execute REPL code
            exec_result = repl.exec(
                code,
                str(iteration),
                error_if_not_finish,
                False,
                budget_forcing_active,
                exec_timeout_override,
            )

            # Complete the span
            span.complete(exec_result=exec_result)

        return ReplIterationResult(
            exec_result=exec_result,
            repls=repls,
            response_content=response_content,
            code=code,
        )

    def _finalize_repl_output(
        self,
        base_output: str,
        iteration: int,
        max_iterations: int,
        cur_recursion_depth: int,
        max_recursion_depth: int,
        max_input_tokens: int | None = None,
        must_exit_warning: str = "",
    ) -> str:
        """Add common suffixes to REPL output: iteration warning, budget status, prompt.

        Args:
            base_output: The formatted REPL output (e.g., with <repl_output> or
                <repl_error>)
            iteration: Current iteration number (0-indexed)
            max_iterations: Maximum number of iterations allowed
            cur_recursion_depth: Current recursion depth
            max_recursion_depth: Maximum recursion depth allowed
            max_input_tokens: Maximum input tokens for the model (for context window warning)
            must_exit_warning: Pre-rendered must-exit warning text from REPL classes

        Returns:
            The finalized output with warnings, budget status, and prompt appended
        """
        output = base_output
        must_exit = False

        # Add last iteration warning if needed
        if iteration >= max_iterations - 1:
            output += "\nNOTE: This is your last allowed REPL interaction.\n"
            must_exit = True

        # Add context window warning if threshold reached
        if (
            self.config.context_window_fraction is not None
            and max_input_tokens is not None
            and self._last_prompt_tokens is not None
        ):
            if (
                self._last_prompt_tokens / max_input_tokens
                >= self.config.context_window_fraction
            ):
                pct = self._last_prompt_tokens / max_input_tokens * 100
                output += (
                    f"\nWARNING: You have used {pct:.0f}% of the model's context window "
                    f"({self._last_prompt_tokens:,} / {max_input_tokens:,} tokens).\n"
                )
                must_exit = True

        # Check budget and add warnings/status
        if self.cost_tracker is not None:
            if self.cost_tracker.llm_cost_budget is not None:
                output += "\n" + self.cost_tracker.get_llm_cost_budget_status() + "\n"
            if self.cost_tracker.llm_calls_budget is not None:
                output += "\n" + self.cost_tracker.get_llm_calls_budget_status() + "\n"
            if (
                self.cost_tracker.is_llm_cost_budget_exhausted()
                or self.cost_tracker.is_llm_calls_budget_exhausted()
            ):
                must_exit = True

        if must_exit and must_exit_warning:
            output += "\n" + must_exit_warning + "\n"

        # Add invoke calls status if jaz.invoke is available (recursion limit not reached)
        if self.cost_tracker is not None and cur_recursion_depth < max_recursion_depth:
            output += "\n" + self.cost_tracker.get_invoke_calls_status() + "\n"

        # Add next iteration prompt
        output += (
            f"\n**Enter your next REPL input (iteration #{iteration + 1}; "
            f"max {max_iterations} iterations allowed):**\n"
        )
        return output

    @overload
    def invoke[ReturnT](
        self,
        task: str,
        *,
        return_type: TypeForm[ReturnT],
        return_validator: Callable[[ReturnT], None] | None,
        repl_input_validator: Callable[[str], None] | None = None,
        libraries: list[Library],
        max_iterations: int,
        max_iterations_buffer: int,
        cur_recursion_depth: int,
        max_recursion_depth: int,
        allowed_imports: list[str],
        task_name: str = "main",
        invoke_id: str,
        parent_invoke_id: str | None,
        parent_repl_iteration: int | None,
        input_descriptions: dict[str, str | None] | None = None,
        **inputs: object,
    ) -> ReturnT: ...

    @overload
    def invoke(
        self,
        task: str,
        *,
        return_type: None,
        return_validator: None,
        repl_input_validator: Callable[[str], None] | None = None,
        libraries: list[Library],
        max_iterations: int,
        max_iterations_buffer: int,
        cur_recursion_depth: int,
        max_recursion_depth: int,
        allowed_imports: list[str],
        task_name: str = "main",
        invoke_id: str,
        parent_invoke_id: str | None,
        parent_repl_iteration: int | None,
        input_descriptions: dict[str, str | None] | None = None,
        **inputs: object,
    ) -> None: ...

    def invoke[ReturnT](
        self,
        task: str,
        *,
        return_type: TypeForm[ReturnT] | None,
        return_validator: Callable[[ReturnT], None] | None,
        repl_input_validator: Callable[[str], None] | None = None,
        libraries: list[Library],
        max_iterations: int,
        max_iterations_buffer: int,
        cur_recursion_depth: int,
        max_recursion_depth: int,
        allowed_imports: list[str],
        task_name: str = "main",
        invoke_id: str,
        parent_invoke_id: str | None,
        parent_repl_iteration: int | None,
        input_descriptions: dict[str, str | None] | None = None,
        **inputs: object,
    ) -> ReturnT | None:
        # TODO: Remove `| None` and overloads once pyright gets better
        # Use dispatcher span for invoke
        enter_event = InvokeEnter(
            invoke_id=invoke_id,
            parent_invoke_id=parent_invoke_id,
            parent_repl_iteration=parent_repl_iteration,
            prompt=task,
            return_type=return_type,
            inputs=inputs,
            max_iterations=max_iterations,
            cur_recursion_depth=cur_recursion_depth,
            max_recursion_depth=max_recursion_depth,
            libraries=libraries,
            task_name=task_name,
        )
        with self.dispatcher.span_invoke(enter_event) as span:
            # Check if hooks want to halt
            if span.ctx.should_halt:
                _raise_halt_errors(span.ctx.halt_errors)

            # Apply any additional libraries from hooks. Deduping by root module
            # keeps prompt docs aligned with the REPL state when a later library
            # intentionally replaces an existing namespace (for example `env`).
            all_libraries = _dedupe_libraries_by_root_module(
                libraries + span.ctx.additional_libraries
            )

            # Get budget status if cost tracker is available AND has a budget
            budget_status = None
            if self.cost_tracker is not None:
                budget_parts = []
                if self.cost_tracker.llm_cost_budget is not None:
                    budget_parts.append(self.cost_tracker.get_llm_cost_budget_status())
                if self.cost_tracker.llm_calls_budget is not None:
                    budget_parts.append(self.cost_tracker.get_llm_calls_budget_status())
                if budget_parts:
                    budget_status = "\n".join(budget_parts)
                    if (
                        self.cost_tracker.is_llm_cost_budget_exhausted()
                        or self.cost_tracker.is_llm_calls_budget_exhausted()
                    ):
                        budget_status += (
                            "\n\nYou MUST immediately exit the REPL by using "
                            "`RETURN ...` or `RAISE ...` "
                            "to avoid further costs."
                        )

            # Generate unique session ID to prevent linecache collisions between
            # sessions
            session_id = uuid.uuid4().hex[:8]

            # Initialize REPLs - main REPL + additional REPLs
            repls: dict[str, REPL] = {}

            # Get REPL configs from Config
            repl_configs = self.config.repl_configs

            # Add REPLs from repl_classes
            for lang, repl_class in self.repl_classes.items():
                # Get config for this language, or empty dict for defaults
                repl_config = repl_configs.get(lang, {})
                repls[lang] = repl_class.initialize(
                    task=task,
                    return_type=return_type,
                    inputs=inputs,
                    libraries=all_libraries,
                    allowed_imports=allowed_imports,
                    session_id=session_id,
                    config=repl_config,
                    repl_exec_timeout=self.config.repl_exec_timeout,
                    forbidden_names=self.config.forbidden_names,
                    forbidden_attributes=self.config.forbidden_attributes,
                    return_validator=return_validator,
                    repl_input_validator=repl_input_validator
                    or self.config.repl_input_validator,
                )

            # TODO: Add REPL lang prefix to each line?
            # Collect REPL-specific prompt contributions
            task_preamble = "\n".join(
                filter(None, (r.get_task_preamble() for r in repls.values()))
            )
            inputs_preamble = "\n".join(
                filter(None, (r.get_inputs_preamble() for r in repls.values()))
            )
            must_exit_warning = "\n".join(
                filter(
                    None,
                    (
                        r.get_must_exit_warning(
                            template_name=self.config.must_exit_warning_template,
                            can_delegate=cur_recursion_depth < max_recursion_depth,
                            max_input_length=self.config.max_input_length,
                            delegate_exec_timeout=self.config.delegate_exec_timeout,
                        )
                        for r in repls.values()
                    ),
                )
            )
            finish_hint = max(
                (r.get_finish_command_hint() for r in repls.values()),
                key=len,
                default="`RETURN ...`",
            )

            user_prompt = get_user_prompt(
                user_prompt_template=self.config.user_prompt_template,
                task=task,
                inputs=inputs,
                return_type=return_type,
                max_input_length=self.config.max_input_length,
                budget_status=budget_status,
                instruction_prompt_additions=span.ctx.instruction_prompt_additions,
                input_truncation_advice_template=self.config.input_truncation_advice_template,
                task_preamble=task_preamble,
                inputs_preamble=inputs_preamble,
                input_descriptions=input_descriptions,
                prefix_ratio=self.config.truncation_prefix_ratio,
            )

            # Apply invoke-level REPL inputs from hooks
            if span.ctx.repl_inputs:
                for repl in repls.values():
                    repl.add_inputs(span.ctx.repl_inputs)

            # Append iteration prompt (and budget/exit warnings if applicable)
            user_prompt = self._finalize_repl_output(
                user_prompt,
                iteration=0,
                max_iterations=max_iterations,
                cur_recursion_depth=cur_recursion_depth,
                max_recursion_depth=max_recursion_depth,
                must_exit_warning=must_exit_warning,
            )

            messages: list[MessageDict] = [
                {
                    "role": "system",
                    "content": get_system_prompt(
                        system_prompt_template=self.config.system_prompt_template,
                        libraries=all_libraries,
                        forbidden_names=self.config.forbidden_names,
                        forbidden_attributes=self.config.forbidden_attributes,
                        allowed_imports=allowed_imports,
                        cur_recursion_depth=cur_recursion_depth,
                        max_recursion_depth=max_recursion_depth,
                        max_iterations=max_iterations,
                        available_repls=list(repls.keys()),
                        add_external_thinking=self.config.add_external_thinking,
                        system_prompt_additions=span.ctx.system_prompt_additions,
                        repl_configs=repl_configs,
                        show_tools_in_prompt=self.config.show_tools_in_prompt,
                        full_docstrings_in_prompt=self.config.full_docstrings_in_prompt,
                    ),
                },
                {"role": "user", "content": user_prompt},
            ]
            for message in messages:
                self.dispatcher.process_message(Message(message=message))

            # Budget forcing: track remaining refusals for early RETURN/RAISE
            budget_forcing_refusals_remaining = self.config.budget_forcing_refusals

            # Look up max_input_tokens for context window warning
            max_input_tokens: int | None = None
            if self.config.context_window_fraction is not None:
                model_info = self.llm_client.get_model_info(self.config.model_config)
                if model_info is not None:
                    max_input_tokens = model_info.get("max_input_tokens")
                if max_input_tokens is None:
                    raise ValueError(
                        f"context_window_fraction is set to {self.config.context_window_fraction} "
                        f"but max_input_tokens could not be determined for model '{self.model}'. "
                        "Either set context_window_fraction to None or use a model recognized by the pricing database."
                    )

            for i in range(1, max_iterations + max_iterations_buffer + 2):
                # Update prehook so nested invokes know which iteration spawned them
                if self.prehook is not None:
                    self.prehook.parent_repl_iteration = i

                # Hard stop: if we've exceeded the iteration + buffer limit,
                # raise immediately without executing the REPL
                hard_stop_error: Exception | None = None
                if i > max_iterations + max_iterations_buffer:
                    hard_stop_error = BudgetExhaustedError(
                        "No return value after "
                        f"{max_iterations + max_iterations_buffer} "
                        "REPL iterations"
                    )
                if (
                    hard_stop_error is None
                    and self.cost_tracker is not None
                    and self.cost_tracker.is_llm_cost_buffer_exhausted()
                ):
                    hard_stop_error = (
                        self.cost_tracker.get_llm_cost_buffer_exhausted_error()
                    )
                if (
                    hard_stop_error is None
                    and self.cost_tracker is not None
                    and self.cost_tracker.is_llm_calls_buffer_exhausted()
                ):
                    hard_stop_error = (
                        self.cost_tracker.get_llm_calls_buffer_exhausted_error()
                    )
                if hard_stop_error is not None:
                    span.complete(
                        result=RaiseResult(output="", exception=hard_stop_error)
                    )
                    raise hard_stop_error

                # Tell the REPL what error to return if the next command should
                # end the REPL session but didn't
                error_if_not_finish = None
                if i >= max_iterations:
                    error_if_not_finish = BudgetExhaustedError(
                        f"Expected REPL input to end with {finish_hint} "
                        "on last REPL iteration."
                    )
                # Check LLM cost budget
                if (
                    error_if_not_finish is None
                    and self.cost_tracker is not None
                    and self.cost_tracker.is_llm_cost_budget_exhausted()
                ):
                    # TODO: In native python repl, we need the stronger condition that the REPL input
                    # actually ends the REPL session.
                    error_if_not_finish = BudgetExhaustedError(
                        f"Expected REPL input to end with {finish_hint} "
                        "when LLM cost budget is exhausted."
                    )
                # Check LLM calls budget
                if (
                    error_if_not_finish is None
                    and self.cost_tracker is not None
                    and self.cost_tracker.is_llm_calls_budget_exhausted()
                ):
                    error_if_not_finish = BudgetExhaustedError(
                        f"Expected REPL input to end with {finish_hint} "
                        "when LLM calls budget is exhausted."
                    )
                # Check context window usage
                if (
                    error_if_not_finish is None
                    and self.config.context_window_fraction is not None
                ):
                    assert max_input_tokens is not None
                    if i == 1:
                        # First iteration: no LLM query has been made yet
                        assert self._last_prompt_tokens is None
                    else:
                        assert self._last_prompt_tokens is not None
                        if (
                            self._last_prompt_tokens / max_input_tokens
                            >= self.config.context_window_fraction
                        ):
                            error_if_not_finish = BudgetExhaustedError(
                                "Context window nearly full: "
                                f"{self._last_prompt_tokens:,} / {max_input_tokens:,} tokens "
                                f"({self._last_prompt_tokens / max_input_tokens * 100:.0f}%). "
                                f"Expected REPL input to end with {finish_hint}."
                            )

                # Budget forcing: determine if we should refuse early RETURN/RAISE
                budget_forcing_active = (
                    budget_forcing_refusals_remaining > 0
                    and i < max_iterations
                    and (
                        self.cost_tracker is None
                        or (
                            not self.cost_tracker.is_llm_cost_budget_exhausted()
                            and not self.cost_tracker.is_llm_calls_budget_exhausted()
                        )
                    )
                )

                # Use dispatcher span for REPL iteration
                iter_enter_event = ReplIterationEnter(
                    invoke_id=invoke_id,
                    iteration=i,
                    max_iterations=max_iterations,
                    cur_recursion_depth=cur_recursion_depth,
                    max_recursion_depth=max_recursion_depth,
                )
                with self.dispatcher.span_repl_iteration(iter_enter_event) as iter_span:
                    # Check if hooks want to halt the entire invoke
                    if iter_span.ctx.should_halt:
                        _raise_halt_errors(iter_span.ctx.halt_errors)

                    # Apply iteration-level REPL inputs from hooks
                    if iter_span.ctx.repl_inputs:
                        for repl in repls.values():
                            repl.add_inputs(iter_span.ctx.repl_inputs)

                    # Execute one REPL iteration
                    repl_iter_result = self.do_one_repl_iteration(
                        i,
                        max_iterations,
                        cur_recursion_depth,
                        messages,
                        repls,
                        error_if_not_finish,
                        invoke_id,
                        budget_forcing_active,
                        instruction_prompt_additions=(
                            iter_span.ctx.instruction_prompt_additions
                            if iter_span.ctx.instruction_prompt_additions
                            else None
                        ),
                    )

                    # Complete the span with the result
                    iter_span.complete(result=repl_iter_result)

                # Unpack the result (response_content and code are in the result but not used here)
                exec_result = repl_iter_result.exec_result
                repls = repl_iter_result.repls

                # Check if budget forcing actually blocked a RETURN/RAISE
                # Only decrement the counter if it did
                if isinstance(exec_result, ErrorResult) and isinstance(
                    exec_result.exception, BudgetForcingException
                ):
                    budget_forcing_refusals_remaining -= 1

                match exec_result:
                    case ExecuteResult():
                        output = exec_result.output
                        if not isinstance(output, str):
                            raise _JazInternalError(
                                "REPL EXECUTE output should be a string"
                            )
                        abbreviated_output, output_truncated = abbreviate_string(
                            output,
                            max_length=self.config.max_repl_output_length,
                            prefix_ratio=self.config.truncation_prefix_ratio,
                        )
                        repl_output_formatted = (
                            f"**REPL output (iteration #{i}):**\n\n"
                            f"<repl_output>\n{abbreviated_output}</repl_output>\n"
                        )
                        # Add truncation advice if output was truncated
                        # TODO: repl_output_truncation_advice_template should be a REPL config not a global config
                        # TODO: Add REPL lang prefix to each line
                        if (
                            output_truncated
                            and self.config.repl_output_truncation_advice_template
                        ):
                            advice = "\n".join(
                                filter(
                                    None,
                                    (
                                        r.get_truncation_advice(
                                            template_name=self.config.repl_output_truncation_advice_template,
                                            iteration=i,
                                            output_truncated=True,
                                            error_truncated=False,
                                        )
                                        for r in repls.values()
                                    ),
                                )
                            )
                            if advice:
                                repl_output_formatted += "\n" + advice
                        repl_output_formatted = self._finalize_repl_output(
                            repl_output_formatted,
                            i,
                            max_iterations,
                            cur_recursion_depth,
                            max_recursion_depth,
                            max_input_tokens=max_input_tokens,
                            must_exit_warning=must_exit_warning,
                        )
                        messages.append(
                            {"role": "user", "content": repl_output_formatted}
                        )
                        self.dispatcher.process_message(Message(message=messages[-1]))
                    case ErrorResult():
                        output = exec_result.output
                        error_summary = exec_result.error_summary
                        if not isinstance(output, str) or not isinstance(
                            error_summary, str | None
                        ):
                            raise _JazInternalError(
                                "REPL ERROR output and error summary should be strings"
                            )
                        abbreviated_output, output_truncated = abbreviate_string(
                            output,
                            max_length=self.config.max_repl_output_length,
                            prefix_ratio=self.config.truncation_prefix_ratio,
                        )
                        if error_summary:
                            abbreviated_error_summary, error_truncated = (
                                abbreviate_string(
                                    error_summary,
                                    max_length=self.config.max_repl_output_length,
                                    prefix_ratio=self.config.truncation_prefix_ratio,
                                )
                            )
                        else:
                            abbreviated_error_summary = ""
                            error_truncated = False

                        # Build the formatted message
                        repl_output_formatted = f"**REPL output (iteration #{i}):**\n\n"

                        # Add error summary if provided
                        if abbreviated_error_summary:
                            repl_output_formatted += (
                                "*An error occurred:*\n"
                                f"<error>\n{abbreviated_error_summary}\n</error>\n\n"
                            )
                        else:
                            repl_output_formatted += (
                                "*An error occurred during execution.*\n\n"
                            )

                        repl_output_formatted += (
                            f"<repl_output>\n{abbreviated_output}</repl_output>\n"
                        )

                        # Add truncation advice if any truncation occurred
                        if (
                            output_truncated or error_truncated
                        ) and self.config.repl_output_truncation_advice_template:
                            # TODO: Should not concatenate; should only take the REPL who gave the output?
                            advice = "\n".join(
                                filter(
                                    None,
                                    (
                                        r.get_truncation_advice(
                                            template_name=self.config.repl_output_truncation_advice_template,
                                            iteration=i,
                                            output_truncated=output_truncated,
                                            error_truncated=error_truncated,
                                        )
                                        for r in repls.values()
                                    ),
                                )
                            )
                            if advice:
                                repl_output_formatted += "\n" + advice
                        repl_output_formatted = self._finalize_repl_output(
                            repl_output_formatted,
                            i,
                            max_iterations,
                            cur_recursion_depth,
                            max_recursion_depth,
                            max_input_tokens=max_input_tokens,
                            must_exit_warning=must_exit_warning,
                        )
                        messages.append(
                            {"role": "user", "content": repl_output_formatted}
                        )
                        self.dispatcher.process_message(Message(message=messages[-1]))
                    case ReturnResult():
                        # REPL handles budget forcing - if we get here, it wasn't active
                        assert not budget_forcing_active, (
                            "ReturnResult should not be returned when budget forcing "
                            "is active - REPL should have returned ErrorResult"
                        )
                        # Complete the invoke span
                        span.complete(result=exec_result)
                        # REPL should have already enforced return type correctness
                        # and returned ErrorResult if not
                        return exec_result.return_value
                    case RaiseResult():
                        # REPL handles budget forcing - if we get here, it wasn't active
                        assert not budget_forcing_active, (
                            "RaiseResult should not be returned when budget forcing "
                            "is active - REPL should have returned ErrorResult"
                        )
                        # Complete the invoke span
                        span.complete(result=exec_result)
                        # REPL should have already enforced raise type correctness
                        # and returned ErrorResult if not
                        # TODO: check that traceback is preserved
                        raise exec_result.exception
                    case _:
                        raise _JazInternalError("Unhandled REPL result type")

            raise _JazInternalError("Unreachable code reached in Agent.invoke")
