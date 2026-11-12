## 0.7.2 — Single terminal, plan lifecycle and context safety — 2026-11-11

- Replaced the model-visible terminal start/poll/input/interrupt/resize tool set with one `terminal` tool. SWE-ReX operation IDs, cursors and background polling are internal manager details.
- The terminal tool waits for completion; it returns control after 10 seconds without new output, and a single call waits no longer than 300 seconds even if output continues. These are return-to-LLM thresholds, not process-kill timeouts.
- The model can resume waiting with an empty terminal call, send interactive input through the same tool, or interrupt the active foreground command with Ctrl+C.
- Returns the accumulated PTY transcript as plain tool-message text without a JSON wrapper or appended exit code. A shell command's nonzero exit is not treated as failure of the LocalAgent tool call; its exit status remains private metadata for deterministic acceptance checks.
- After terminal completion/interruption/timeout/failure, probes the persistent Bash session through `create_session`; if the session disappeared it recreates it and tells the model that shell variables/functions were lost. SWE-ReX output truncation is surfaced rather than silently presented as complete.
- Added `TERMINAL_IDLE_TIMEOUT_SECONDS` and `TERMINAL_MAX_WAIT_SECONDS` settings and regression tests for the single-tool contract.
- Removed automatic Direct-to-Plan escalation after repeated tool failures. After each two additional consecutive failures, the controller asks the model to reconsider; only an explicit model choice switches to Plan.
- New plans no longer assign a mandatory `completion_mode`. Ordinary stages may be completed from already collected facts and model review; only an explicit machine-checkable `acceptance_check` is a hard verification gate. Legacy `completion_mode` fields remain readable but are not enforced.
- Treat `skipped` as a resolved stage in completion, dependency and plan-revision logic. Replaced the stage-review `blocked` decision with `question`; a user question pauses the task and is resumed through the existing clarification flow.
- Explicitly prohibit final-answer/final-report stages in generated plans.
- Context budgeting now includes the tool schema and response reserve. At 50% it only deduplicates by stable event ID; at 60% it records a deterministic growth forecast; at 70% it asks the LLM to compact a factual projection while retaining the original transcript archive.
- Added conservative context preflight for `load_file_for_model`, plus recovery from actual Ollama context-overflow errors by removing the failed payload from the next request and returning a clear error to the model.
- Added regression coverage for plan choice, skipped/question lifecycle, context compaction and direct-file-load overflow.

## 0.7.1 — Background terminal, cancellation, proxy lifecycle and trace — 2026-11-10

- Routed the default `terminal` tool through SWE-ReX background operations with a one-second quick-result window, idempotency key, incremental cursor and explicit polling/input/interrupt tools. Process timeout is unlimited by default unless explicitly supplied. Preserved the legacy synchronous manager method for compatibility.
- Normalized SWE-ReX operation output, completion state, exit code, timeout, interruption, duration, truncation and cursor without claiming separate stdout/stderr when SWE-ReX supplies a combined output stream.
- Added an independent cancellation barrier: Ollama stream closure and remote terminal interruption run outside the task worker; the task is not reported cancelled until the remote control result is terminal/idle, otherwise it remains interrupted with a warning.
- Added shared Xray lifecycle management: reuse the existing named container, start it when stopped, attach it to the configured sandbox network, single-flight checks, and bounded DNS/SOCKS5/Privoxy egress probing from inside the sandbox. Network-dependent terminal commands require a successful probe and get one controlled recovery/recheck; recognized proxy/DNS failures during a running command trigger a health recheck but never replay that command. The manager never removes the shared proxy.
- Added opt-in SQLite Debug Trace events for model requests/responses, returned thinking (only when separately enabled), tool calls/results/errors and controller checkpoints; included recursive secret redaction, per-task cascade deletion, JSON export and UI controls.
- The active Ollama model now selects Direct vs Plan for new tasks; invalid/unavailable decisions fall back to the conservative classifier. Context maintenance assigns stable event IDs, removes only repeated IDs at about 50%, records a model-aware growth forecast at about 60%, and compacts the working history at about 70% while retaining the raw checkpoint archive. Summary compaction remains deterministic; LLM rewriting is not enabled until preservation can be validated safely.
- Registered and exposed `load_file_for_model` as required; updated conflicting legacy tool-catalog tests and documentation.
- Updated cancellation, proxy, trace, tool catalog and five-state plan regression coverage.

## 0.7.0 — Sandbox, runtime integration and data integrity — 2026-11-10

- Added safe SQLite foreign-key migrations for task-scoped tool and model journals.
- Updated release metadata and architecture documentation to 0.7.0.
- Further runtime, proxy, cancellation and orchestration changes are tracked in this release patch.

## 0.6.2 — Sandbox deletion ID matching — 2026-11-08

- Fixed sandbox deletion failing when Podman returned an abbreviated container ID while SQLite stored the full ID; container discovery now requests untruncated IDs.
- Added a regression test that simulates Podman's default ID truncation and verifies deletion uses the full ID.

## 0.6.1 — Terminal timeout and tool-call recovery — 2026-11-08

- Added host-side Bash syntax-only validation before dispatch, preventing unmatched quotes and malformed heredocs from blocking the persistent shell.
- Added bounded SWE-ReX Bash-session reset and readiness probe after command timeouts; the conversation container and filesystem are preserved.
- Exposed explicit shell recovery/reset state in timeout results and mark the sandbox unhealthy when recovery cannot be confirmed.
- Made Ollama tool-call argument normalization strict and checkpoint-safe: malformed arguments become recoverable tool-contract errors rather than aborting the task loop.
- Updated terminal/system instructions and architecture documentation with timeout and shell-reset semantics.

# Changelog

## Planner reliability fixes — 2026-11-08

- Fixed structured terminal error classification: `timed_out: false` is no longer treated as a timeout; non-zero exit codes are classified as execution/verification failures.
- Preserved successful tool receipts when execution escalates from the simple strategy to a newly generated multi-step plan. Migrated receipts are explicitly marked as task-level history, not proof that a particular stage is complete.
- Added reviewer-driven reconciliation for useful work completed ahead of the current plan stage. Only existing future step IDs, successful tool evidence, dependency order and any configured deterministic acceptance checks are accepted.
- Added regression tests for timeout classification and idempotent evidence migration.

## 0.6.0 — Rootless Podman + SWE-ReX sandbox

- Introduced one persistent rootless Podman Linux container per conversation.
- Added Sandbox Manager, SWE-ReX 1.4.0 RemoteRuntime adapter and persistent Bash session `main`.
- Added SQLite sandbox metadata migration and container restoration after LocalAgent restart.
- Replaced exposed file/Python/Office/SQLite operation tools with a universal terminal tool.
- Preserved web search/page tools and added explicit file publishing for user downloads.
- Legacy Bubblewrap code remains inactive in the source tree until the new runtime is validated; it is not exposed to the model or selected as an automatic fallback.
- The 10 GB disk quota is not implemented.


## 0.5.6 — 2026-11-06

- Rebuilt compacted working context from the durable raw archived transcript and current archived tool observations. Previous LLM-authored summaries are no longer recursively fed back into future summaries; free-form assistant prose is not promoted to facts or instructions.
- Removed the extra LLM context-selection call after compaction. This avoids hallucinated/stale `next_actions` becoming persistent instructions and reduces one model request per compaction.
- Re-inject the live plan from checkpoint after the factual history projection, including concise stage statuses and verified results. The history summary is explicitly labelled as observations, not a new plan.
- Replaced rigid stage prohibitions with flexible current-focus guidance. Efficient adjacent/overlapping work is allowed; meaningful work beyond the current focus is reported for controller reconciliation.
- New tasks use a deterministic concise per-stage instruction instead of an additional LLM prompt-generation call. Existing checkpoints that explicitly enabled prompt generation remain compatible.
- Missing-file/read errors can be cleared by a later successful mutation of the exact same path; unrelated successful actions do not clear them. Failure checkpoints now retain only safe path references needed for this reconciliation, not file contents.
- No tool catalog, sandbox implementation, SQLite schema, Ollama request options, or tag-derived runtime parameters changed.


## 0.5.5 — 2026-11-05

- Ollama model tags are now the sole source of truth for runtime generation parameters. `OllamaClient.chat()` no longer sends a per-request `options` object, so Modelfile values such as `num_ctx`, `num_gpu`, `num_predict` and `temperature` are not overwritten by LocalAgent settings.
- Per-tag context discovery and task checkpointing remain for LocalAgent history budgeting and compaction only; they do not alter Ollama model launch parameters. The agent no longer invents a `num_predict` limit from half the context; it records that value only when explicitly present in the tag.
- Added regression tests ensuring chat requests do not override model-tag parameters.

## 0.5.4 — 2026-11-04

- Added an explicit current-stage execution boundary to the planner, step-task generator and live plan context. The executor must stop after the current stage; only the Python controller advances the plan.
- Step-task generation now receives later stages as explicit out-of-scope work and must define a stop condition.
- Added conservative workspace reconciliation after stage completion. Future stages are auto-completed only when their own deterministic acceptance checks pass against actual workspace state; stages without checks are never inferred complete.
- Reconciliation decisions are checkpointed. No tool catalog changes or SQLite schema migration.
- This is a first-stage gate, not a hard semantic sandbox: it does not classify and block every tool action that might belong to a later stage.


## 0.5.3 — 2026-11-03

- Planning uses a concise numbered/bulleted text outline instead of requiring the model to emit a strict multi-field JSON plan. Legacy JSON planner responses remain supported; Python assigns ids and conservative defaults.
- Planning instructions favor a small number of meaningful result-oriented stages and defer low-level decomposition until a stage becomes current.
- Step review and forced replanning receive a compact whole-task planning snapshot: original goal, all stage statuses, attempt counts, recent dialogue and prior plan revisions.
- Replanning explicitly reassesses the entire remaining route to the original goal, preserves completed work and rejects an exactly unchanged remaining-step list.
- Added a bounded plan-revision history to the task checkpoint. No SQLite schema migration or changes to tools are required.

## 0.5.2 — 2026-11-01

- Ollama model context is resolved per exact model tag: explicit Modelfile `num_ctx` wins; conventional `-32k`/`-64k` tag suffixes are supported; `OLLAMA_NUM_CTX` is a fallback only. Fallback profiles are not cached.
- Each task pins its model, context window and `num_predict` in its checkpoint. `num_predict` is 50% of that task model's context. Context compaction and usage percentages use the same pinned context.
- Ollama chat uses streaming responses. The web UI displays an in-progress approximate generation rate and, after completion, Ollama's exact generation rate plus prompt-processing rate, model and context details.
- Deterministic acceptance checks with successful tool evidence can complete a step without a second LLM review call. The worker's already-generated final answer is delivered; a safe fallback is used if it is empty.
- Added regression coverage for model-profile resolution, task-pinned context, streaming output and token-rate statistics.


## 0.5.1 — 2026-11-01

- Python sandbox can write to both the shared `/workspace` and task-scoped `/work`; network isolation and mount boundaries remain in place. `copy_file` remains the preferred verified transfer tool, while Python copy operations are permitted.
- `num_predict` is derived dynamically as 50% of configured `num_ctx`; the obsolete fixed `OLLAMA_NUM_PREDICT` setting is removed from the example configuration.
- All registered tools are exposed on every turn. Tool scope messages group them by action category instead of hiding tools with regex routing. Removed duplicate public `sqlite_read`; `sqlite_query` covers read-only queries. Legacy dispatch remains for saved checkpoints.
- Unified path conventions: `/workspace/path` for shared files, `/work/path` for task files, relative paths for `/workspace`; file, Excel, Word and database tools resolve both namespaces.
- Added model-direct audio input for supported Ollama multimodal models, including Gemma 4 12B; uploaded audio is normalized to mono 16 kHz WAV (30-second limit). Video analysis sends sampled frames and, when present, up to 30 seconds of soundtrack audio.
- Rewrote agent instructions, planning, step prompts, review and tool routing guidance in English; user-facing replies continue in the user's requested language.
- Added live generation speed in the web header (`tok/s`) using Ollama `eval_count` and `eval_duration`; fixed client model state synchronization.
- Improved JSON parsing for fenced/embedded JSON and trailing commas; an invalid planner response falls back to a single direct-execution step instead of failing before work begins. Simple one-file requests bypass the full planner/reviewer chain.
- Added regression tests for writable workspace sandbox, audio handoff, dynamic output limit, tool visibility, duplicate removal, model selection and generation speed.


## 0.5.0 — 2026-11-01

- Выбор модели Ollama хранится в SQLite `app_settings`; `.env` и `OLLAMA_MODEL` не перезаписываются.
- Модель закрепляется в checkpoint каждой новой задачи и применяется при возобновлении.
- Строгая схема `run_python`: ровно один из `code`/`path`, с реальной проверкой `oneOf` и `not` до dispatch.
- Сохранены исправления 0.4.10: `copy_file`, резервное копирование при перезаписи, нормализация истории Ollama и защита коротких Python-workflow.

## 0.4.10 — 2026-10-31

- Исправлено продолжение диалога после ответа модели: перед новым ходом гарантируется корректное сообщение пользователя; соседние текстовые сообщения assistant безопасно объединяются, некорректные tool-call последовательности не маскируются.
- В Python-сценариях `copy_file` доступен всегда; системные инструкции и список инструментов согласованы. Публикация объединена с `copy_file`, внутренний legacy-dispatch сохранён для старых checkpoint.
- `copy_file` сохраняет резервную копию при перезаписи файла workspace из `/work`, проверяет SHA-256 и результат копирования.
- Убрана автоматическая выдача модели дублирующего `publish_workspace_file`.
- Добавлен выбор установленной модели Ollama в веб-интерфейсе. Список берётся из `/api/tags`; выбор сохраняется в `.env`; переключение блокируется при активных/приостановленных задачах.
- Версия: 0.4.10.

## 0.4.9 — 2026-10-31

- Исправлена передача `task_id` всем инструментам с аргументами `*_path` для псевдонима `work/...` и `/work/...`. Это охватывает файловые, Excel- и SQLite-инструменты и устраняет ошибку `The /work path requires the current task context` при работе в области задачи.
- Исправлены рекомендации и диагностика для коротких Python-задач: скрипты из `/work` запускаются в task-scoped каталоге, а все создаваемые скриптом файлы должны сохраняться в `/work`, а не в read-only `/workspace`.
- Настройки Ollama вынесены в конфигурацию: модель `ornith-1.5:9b`, `num_ctx=65536`, `num_gpu=99`, `num_predict=4096`, `temperature=0.7`. Параметры передаются в каждом запросе `/api/chat`, включая планирование, основной цикл и проверку этапов.
- `num_gpu=99` запрашивает максимально возможное число GPU-слоёв, но не гарантирует полную выгрузку на видеокарту при нехватке VRAM. Фактическое распределение проверяется через `ollama ps`; на RX 6650 XT контекст 64K может приводить к частичному CPU offload.
- Добавлены регрессионные тесты псевдонима `/work` и полного набора параметров Ollama. Схема SQLite не меняется; `AUTO_RESUME_TASKS` не изменяется.

# Changelog

## 0.4.8 — 2026-10-30

- Unified virtual path handling for `/workspace/...`, workspace-relative paths and task-scoped `/work/...` across file operations, Excel/Word/SQLite helpers, file loading and Python script resolution.
- `/work` text writes overwrite an existing task-working file by default; workspace writes remain protected unless `overwrite=true`.
- Removed the stale static tool catalogue from the system prompt. The per-turn tool scope is generated from the canonical `TOOL_DEFINITIONS` registry; an explicit request to describe all tools receives the full registry, while ordinary tasks receive a focused subset.
- Planner guidance now forbids inventing filenames/paths and preserves user-specified paths. `/workspace/...` is also supported in acceptance-check resolution; acceptance checks no longer force an invented workspace filename.
- Added per-request Ollama token telemetry in SQLite: prompt/eval counts and durations, per-task totals, current/max prompt occupancy against configured `num_ctx`, and a web UI summary.
- Preserved existing database rows; telemetry table is created idempotently during initialization.


## 0.4.6 — 2026-10-30

- Ускорена обработка коротких автономных задач вида «создай Python-скрипт в /work, запусти и проверь»: они обходят отдельный вызов планировщика и генерацию инструкций для каждого этапа, сохраняя обычный цикл tool calling, журнал операций и контроль ошибок.
- Для явно запрошенного Python-сценария сужен набор инструментов до записи/чтения файлов и запуска Python; нерелевантные Excel/SQLite-инструменты не предлагаются одновременно с Python-реализацией.
- Для такого сценария успешное завершение требует подтверждённых вызовов `write_file` и `run_python`; отсутствие любого из них блокирует ложный статус `completed`.
- Исправлена блокировка полезного повтора: если `run_python` не нашёл `/work/script.py`, а затем успешный `write_file` создал именно этот путь, повтор запуска разрешается. Успешная запись другого файла по-прежнему не снимает блокировку. Аналогично обрабатываются некоторые другие ошибки отсутствующего файла.
- Результат `write_file` в `/work` теперь сообщает фактический task work directory; ошибка отсутствующего скрипта в `run_python` включает task_id, разрешённый каталог и список видимых файлов для диагностики рассинхронизации путей.
- Добавлены регрессионные тесты маршрутизации, сужения инструментов и передачи одного task_id через запись/запуск. Схема SQLite и пользовательские данные не меняются.


## 0.4.5 — 2026-10-30

- Простые вопросы по прикреплённым изображениям и документам маршрутизируются непосредственно в Ollama без отдельного вызова планировщика; для такого хода инструменты отключены.
- Перед каждым рабочим ходом в системный контекст добавляется точный список доступных инструментов, совпадающий с переданными JSON-схемами. Это устраняет противоречие между общим описанием возможностей и фактическим набором инструментов.
- Добавлен `load_file_for_model`: файлы из workspace и `/work` передаются в следующий контекст Ollama; изображения используют нативный `images`, текстовые файлы и извлечённое содержимое PDF/DOCX/XLSX поступают как текст.
- Добавлен `copy_file` для безопасного копирования между workspace и `/work`, с запретом перезаписи по умолчанию, проверкой SHA-256, защитой от path traversal и символических ссылок.
- Финализация блокируется, если в состоянии задачи остаются неподтверждённые ошибки инструментов; задача переводится в `paused`, а не в ложный `completed`.
- Обновлены документация и регрессионные тесты. Миграция SQLite и изменение `.env` не требуются.

## 0.4.4 — 2026-10-29

- Исправлена проверка `acceptance_check.path`: разрешены безопасные пути `/work/...` для файлов и скриптов текущей задачи. Проверки разрешают путь только внутри рабочего каталога конкретной задачи; `..` и символические ссылки, ведущие за пределы каталога, не проходят.
- Проверка `file_exists`, `file_contains`, `directory_exists`, `python_exit_code` и `python_tests_passed` теперь понимает виртуальные пути `/work/...`; при восстановлении задачи передаётся её `task_id`.
- Уточнён системный промпт планировщика: промежуточные файлы можно проверять в `/work`, а результаты, которые должны остаться пользователю, следует публиковать в `workspace` через `publish_workspace_file`.
- Добавлены регрессионные тесты на планирование, проверку файлов/скриптов в `/work`, обход путей и символьные ссылки.

## 0.4.3 — 2026-10-29

- Исправлена работа с файлами текущей задачи: `write_file`, `read_file`, `edit_file`, `create_directory`, `list_files`, `move_file`, `delete_file` и `delete_directory` поддерживают явный префикс `/work/`; пути разрешаются только внутри каталога текущей задачи.
- `run_python(path="/work/имя.py")` запускает скрипт, созданный в рабочем каталоге задачи. Обычные относительные пути workspace по-прежнему доступны только для чтения внутри Bubblewrap.
- Исправлена ошибка согласования инструментов: текущий `task_id` передаётся в файловые инструменты, чтобы `/work` указывал на тот же каталог между вызовами.
- Усилена защита от циклов: посторонний успешный вызов (например, `list_files`) больше не снимает блокировку точного вызова, который ранее завершился ошибкой. Повтор разрешается после успеха именно этого вызова; диагностические сообщения после нескольких одинаковых ошибок настойчивее требуют сменить стратегию.
- Обновлены системные инструкции и документация: для многошагового Python-сценария агент может создать `/work/script.py`, прочитать/изменить его и запустить через `run_python`.
- Добавлены регрессионные тесты изоляции `/work`, запрета path traversal и symlink escape, запуска созданного скрипта и повторных ошибок.

## 0.4.2 — 2026-10-29

- `run_python` принимает `code` для одноразового выполнения или `path` для существующего скрипта; код запускается в прежней Bubblewrap-песочнице без сети и без shell-инструмента.
- Добавлен `excel_aggregate`: объединение выбранных листов по заголовкам, группировка, сумма/среднее числовых столбцов и проверка созданной книги.
- Добавлен простой read-only инструмент `sqlite_read`; расширенный `sqlite_query` оставлен для задач, где действительно нужны схемы, фильтры, агрегаты и диагностика.
- Перед каждым вызовом модели LocalAgent выбирает релевантный набор инструментов по задаче/этапу; проверка вызова выполняется относительно реально предоставленного набора.
- Автоматический выбор стратегии: простые задачи обходятся без отдельного вызова планировщика и подготовки инструкции этапа; сложные задачи сохраняют плановый цикл. После повторных ошибок простая задача пытается перейти к плановой стратегии.
- Добавлены критерии маршрутизации задач и документация о поведении простого/планового режима.
- Добавлена загрузка до 5 изображений и документов в веб-чате. Изображения передаются Ollama через нативное поле `images`; PDF/DOCX/XLSX и текстовые файлы извлекаются в ограниченный текстовый фрагмент. Данные вложений хранятся с метаданными в SQLite, миграция выполняется автоматически.
- Улучшены подсказки о выборе инструментов и диагностика для Excel/SQLite.
- Версия синхронизирована в `VERSION`, `agent/version.py`, README и архитектурной документации.

## 0.4.1 — 2026-10-29

- Добавлена программная блокировка точного повторения вызова инструмента после зарегистрированной ошибки; в диагностике сохраняется fingerprint вызова и предыдущая ошибка. После успешного действия блокировка снимается, чтобы разрешить повтор после изменения состояния/получения новых данных. При возобновлении старой задачи применяется актуальная системная инструкция без сброса плана и истории.
- Уточнены стратегии Excel в системном промпте, планировщике и подготовке этапов: сначала обзор листов, затем выборочная обработка, импорт только конкретного листа за вызов, проверка существующих таблиц и запрет слепого повторного импорта.
- Уточнены контракты и описания `read_excel` и `sqlite_import_excel`, включая поведение при отсутствии `sheet_name`.
- Уточнены правила путей для Python-песочницы: скрипт размещается относительным путём в `workspace`, а выходные файлы создаются в `/work` и публикуются отдельным инструментом.
- Обновлены README, архитектурная документация, версия приложения и тесты.
- Добавлены тесты fingerprint/блокировки повторных вызовов.


## 0.4.0 — 2026-10-28

### Единая архитектура исполнения
- Объединены планирование и автономное выполнение: теперь все новые и возобновляемые задачи проходят через один плановый action-observation цикл.
- Один цикл отвечает за составление плана, выбор и вызов инструментов, наблюдение за результатами, проверку этапов, acceptance checks, адаптивную перепланировку, паузу и восстановление.
- Удалено ветвление исполнения по `AGENT_MODE`; параметр больше не читается конфигурацией и не влияет на выполнение. Если он остался в пользовательском `.env`, он игнорируется и его можно удалить вручную. Установщик не изменяет `.env`.
- Новые задачи сохраняют `mode="unified"` и checkpoint schema version 3. Старые контрольные точки `legacy`/`autonomous` загружаются через единый цикл; сохранённые сообщения и квитанции инструментов сохраняются, а при отсутствии валидного плана создаётся совместимый одношаговый план без дополнительного вызова планировщика для старых сообщений.
- `_run_task` — единая точка входа. Старое имя `_run_autonomous_task` оставлено только как совместимый вызов того же runner, не как отдельная реализация цикла.
- README и архитектурная документация обновлены, добавлены тесты единого режима и общего entry point.

### Сохранённые гарантии
- Схема SQLite и файлы пользователя не меняются; миграция базы не требуется.
- Проверка контрактов инструментов, журнал операций, структурированная диагностика, состояние исполнения и события из 0.3.8 остаются включены.
- Планирование, проверка этапов и адаптивная перепланировка не отключаются переключателем режима.
- `AUTO_RESUME_TASKS=false`, Ollama, ComfyUI, ROCm, `.env` и зависимости не изменяются установщиком.


## 0.3.8 — 2026-10-28

### Управление выполнением и восстановление
- Добавлен `agent/orchestration.py`: единый программный контракт вызова инструментов, классификация ошибок, политика безопасного повтора и компактное состояние исполнения.
- Каждый вызов инструмента проверяется против зарегистрированной JSON-схемы до диспетчеризации: имя, обязательные/лишние поля, типы, enum и числовые/строковые границы. Проверка не заменяет ограничения безопасности самих инструментов.
- Диагностика ошибок теперь дополнительно возвращает категорию восстановления, тип эффекта инструмента и указание, безопасно ли повторять действие без проверки состояния. Существующие поля диагностики сохранены для совместимости.
- В checkpoint сохраняются счётчики успешных/неудачных действий, категории ошибок, последние ошибки и ограниченная история событий. Эти сведения восстанавливаются вместе с задачей и не требуют миграции SQLite.
- Добавлен `GET /api/tasks/{tid}/execution`; интерфейс показывает число успешных/неудачных действий, события выполнения и сводку плана, если он есть.

### Планирование и проверка завершения
- Для новых задач режим по умолчанию использует существующий плановый цикл с проверкой этапов и адаптивной перепланировкой. Явное `AGENT_MODE=autonomous` сохраняет прежний action-observation loop. Уже запущенные/приостановленные задачи продолжают режим, записанный в их checkpoint.
- Существующая проверка завершения и acceptance checks сохранены: успех отдельного вызова инструмента сам по себе не подтверждает достижение цели.

### Совместимость и безопасность
- Не меняются `.env`, схема SQLite, рабочие документы, настройки Ollama/ComfyUI и зависимости. `AUTO_RESUME_TASKS=false` не меняется. Git для установки не нужен.
- Добавлены изолированные тесты контрактов инструментов, классификации ошибок и состояния исполнения.

## 0.3.7 — 2026-10-28

- Добавлен инструмент `self_test_tools`: воспроизводимая проверка инструментов на уникальных временных данных с результатами `passed`, `failed` и `not_tested`.
- Добавлен API `POST /api/tools/self-test` и кнопка «Проверить инструменты» в веб-интерфейсе.
- Ужесточена проверка завершения: запросы проверить все инструменты требуют фактического отчёта самопроверки с отдельным статусом каждого инструмента; непроверенные инструменты нельзя выдавать за успешно проверенные.
- Уточнены инструкции автономного цикла: выполнять прямые запросы на действие, использовать программную проверку инструментов и честно сообщать о непроверенных пунктах.
- Уменьшено периодическое перерисовывание интерфейса: список задач, отчёт и история операций обновляются только при изменении данных.
- Панели «Работа агента» и «Задачи и управление» получили фиксированную высоту и внутреннюю прокрутку, чтобы периодические обновления не сдвигали чат.
- Версия в `VERSION`, `agent/version.py` и документации синхронизирована на `0.3.7`.

## 0.3.6 — 2026-10-28

### Исправление панели прогресса
- Панель прогресса по умолчанию свёрнута и занимает одну компактную строку, не вытесняя сообщения диалога.
- Добавлена кнопка «Развернуть / Свернуть» с доступным состоянием `aria-expanded`.
- При раскрытии высота панели, истории вызовов и отчёта ограничена; длинный отчёт прокручивается внутри панели.
- История вызовов инструментов по умолчанию закрыта и открывается отдельно.
- Удалено дублирование заголовка текущей работы внутри панели: строка статуса остаётся рядом с заголовком.

## 0.3.1 — 2026-10-27

### Excel: чтение больших книг и диапазонов
- Лимит входных Office-файлов увеличен с 5 до 50 MB; лимит распакованного содержимого OOXML — до 200 MB, отдельного внутреннего элемента — до 100 MB.
- `read_excel` теперь принимает `start_row` и возвращает размеры листа, диапазон прочитанных строк, `next_start_row` и `has_more_rows` для последовательного чтения чанками.
- Вызов без `sheet_name` даёт компактный обзор имён/размеров листов и короткие превью; для полного чтения нужно выбрать лист и пройти все чанки.
- Ответ `read_excel` ограничен примерно 10 000 ячейками и 80 000 символами, чтобы не переполнять контекст модели.

### Подсказки по выбору инструмента
- Системный и планирующий промпты мягко направляют модель к специализированным инструментам Excel/Word/веб-страниц вместо ручного разбора внутренних форматов, когда специализированный инструмент подходит.
- Описания `read_excel` и `write_excel` уточняют назначение, ограничения и предпочтительный сценарий применения. Это подсказка, а не жёсткое принудительное перенаправление.

### Совместимость
- Схема SQLite и `.env` не меняются. `AUTO_RESUME_TASKS`, лимит итераций и остальная конфигурация пользователя не меняются.
- Чтение Excel по-прежнему не исполняет формулы; поддерживается `.xlsx`, не `.xls`/`.xlsm`.

## 0.3.5 — 2026-10-28

### SQLite-инструменты
- Добавлен `sqlite_query`: список таблиц, описание схемы, структурированные фильтры/сортировка, агрегаты и произвольные read-only запросы с параметрами.
- Добавлен `sqlite_create_database` для создания пользовательской базы без перезаписи существующей.
- Добавлен `sqlite_execute` для изменения данных/схемы пользовательских баз с параметризацией значений, транзакцией и настраиваемым тайм-аутом.
- Добавлены потоковые `sqlite_import_excel` и `sqlite_export_excel` для обработки больших `.xlsx` через SQLite без загрузки всей таблицы в контекст модели.
- Служебная база LocalAgent доступна через `sqlite_query(database_path="agent")` только для чтения. Пользовательские базы ограничены workspace; SQL authorizer запрещает ATTACH/DETACH, PRAGMA, SQL-управление транзакциями и загрузку расширений.
- Результаты SQLite ограничиваются объёмом текста, по умолчанию 24 000 символов, и поддерживают пагинацию; тайм-аут можно настраивать до 600 секунд.

### Прогресс и отчёт при паузе
- Добавлен API `GET /api/tasks/{tid}/operations` с краткой историей вызовов инструментов.
- Под диалогом показываются текущее действие, вызовы инструментов, результаты и ошибки; дублирующая панель над сообщениями убрана.
- При паузе API формирует промежуточный отчёт по журналу операций и контрольной точке: успешные вызовы, ошибки, неподтверждённые операции и оставшиеся шаги плана.
- Сохранены настройки `.env`, `AUTO_RESUME_TASKS=false`, рабочие документы и существующая база данных; миграция схемы для этой функции не требуется.

### Проверки
- Добавлены тесты SQLite-баз, фильтров, агрегатов, параметризации, ограничений доступа, импорта/экспорта Excel, истории операций и отчёта при паузе.
- Полный набор тестов в среде подготовки: 181 passed. В этой среде отсутствует пакет `ddgs`, поэтому для тестового запуска использовалась временная заглушка только в окружении тестов; она не включена в релиз. На целевой машине следует выполнить тесты в `.venv` с установленными `requirements.txt`.

## 0.3.4 — 2026-10-28

### Python-песочница и работа с файлами
- `run_python` теперь использует постоянную рабочую область `/work`, отдельную для каждой задачи и общую для её запусков. `/workspace` остаётся read-only, `/output` хранит журналы конкретного запуска.
- Тайм-аут `run_python` увеличен до 120 секунд (по умолчанию 60); лимит отдельного создаваемого файла увеличен до 100 MiB, лимит виртуального адресного пространства — до 2 GiB.
- Добавлен `publish_workspace_file`: безопасная публикация проверенного файла из `/work` в `workspace`; существующий файл не заменяется без явного `overwrite=true`, перед заменой создаётся резервная копия.
- Обновлены системные инструкции агента для работы через `/work` и публикации проверенных результатов.

### Журналирование и интерфейс
- Все вызовы инструментов теперь получают записи `tool_operations`, включая `run_python`, чтение/поиск и публикацию; известные ошибки тоже сохраняются как завершённые результаты. Незавершённые вызовы остаются неоднозначными и не запускаются автоматически повторно.
- Журнал хранит JSON-сводку аргументов вызова. Большие поля с содержимым документов/таблиц не копируются целиком: сохраняются тип, размер сериализованного значения и SHA-256, чтобы не раздувать базу и не дублировать содержимое файлов.
- `init_db()` автоматически добавляет `arguments_json` в существующую таблицу `tool_operations` без удаления старых записей.
- Восстановление задач умеет восстанавливать результат любого инструментального вызова по записи журнала, а не только `write_file`.
- Добавлено отдельное поле «Текущая работа агента»: показывает реальный вызов инструмента, путь/файл и краткий фрагмент результата или ошибки.
- Обновлены README и архитектурная документация.

### Ограничения
- Python по-прежнему не имеет сетевого доступа и не может изменять `/workspace` напрямую. Для выдачи результата используется отдельный инструмент публикации.
- Рабочие каталоги задач сохраняются в `workspace/.agent_runs/task-<id>/work`; они не очищаются автоматически, чтобы не потерять промежуточные результаты.

# Журнал версий LocalAgent

Все новые выпуски фиксируются здесь. Неизвестная история до первого документированного выпуска не восстанавливается задним числом.

## 0.3.0 — 2026-10-27

### Новые возможности
- Добавлены `read_excel` и `write_excel` для ограниченного чтения и записи диапазонов в `.xlsx` внутри `workspace/`.
- Добавлены `read_word` и `write_word` для чтения/создания `.docx`, включая простые таблицы.
- Добавлен `open_browser_page`: headless Chromium через Playwright для чтения отрисованного текста публичных страниц.
- Добавлен endpoint `GET /api/version` и единый источник версии в `agent/version.py`.
- Добавлены тесты офисных инструментов, ограничения диапазонов и проверки URL.

### Архитектура и безопасность
- Все пять новых инструментов зарегистрированы в `TOOL_DEFINITIONS` и подключены к `dispatch_tool`.
- Офисные файлы доступны только внутри workspace; ограничены размер входного файла, число строк/столбцов/ячеек и объём текста.
- Запись Word защищена от перезаписи существующего файла без `overwrite=true`; запись Excel защищает непустой целевой диапазон без `overwrite=true`.
- Строки, записываемые в Excel, принудительно сохраняются как текст, чтобы входные строки вида `=...` не превращались в формулы.
- Браузер допускает только GET/HEAD, проверяет каждый перехваченный URL, блокирует загрузки, изображения, медиа, шрифты и WebSocket, а также ограничивает число запросов.
- Проверка URL для HTTP-инструментов допускает только стандартные порты 80/443 и запрещает адреса, которые не классифицируются как глобальные публичные IP.
- Веб-страницы обозначены в системном промпте как недоверенные данные, а не инструкции.

### Файлы
- `agent/tools.py` — URL-проверки, офисные инструменты, браузер, схемы инструментов и диспетчеризация.
- `agent/core.py` — описания новых возможностей в системном промпте.
- `agent/web.py` — версия приложения и `GET /api/version`.
- `agent/version.py` — единая версия `0.3.0`.
- `VERSION` — версия релиза.
- `requirements.txt` — зависимости `openpyxl`, `python-docx`, `playwright`.
- `README.md` — запуск, возможности и установка Chromium.
- `docs/ARCHITECTURE.md` — архитектурная документация.
- `tests/test_office_browser_tools.py` — новые тесты.
- `CHANGELOG.md` — журнал изменений.

### Совместимость и ограничения
- Схема SQLite и существующие таблицы базы данных не изменяются.
- Не меняются `.env`, рабочие документы, файлы в `workspace/`, история диалогов и состояние задач.
- Браузер только читает страницы: не нажимает кнопки, не заполняет формы и не отправляет сообщения.
- Для браузера нужен пакет Playwright и отдельная установка Chromium: `python -m playwright install chromium`.
- DNS rebinding не может быть полностью исключён проверкой URL в приложении; для строгой изоляции потребуется сетевой уровень защиты.
- Excel-инструмент читает/записывает `.xlsx`, но не `.xls`, `.xlsm` и не выполняет формулы. Word-инструмент поддерживает `.docx`, не старый `.doc`.
- Для этого выпуска тесты необходимо прогнать в целевом `.venv`; результат из среды подготовки не заменяет проверку на машине пользователя.

### Откат
Использовать `rollback_patch.sh`, созданный при установке патча, с каталогом резервной копии, напечатанным `apply_patch.sh`. Скрипт откажется откатывать файлы, если после установки они были дополнительно изменены.

## Предыдущая история
До `0.3.0` достоверные записи об изменениях в этом журнале отсутствуют. Предыдущие архивы патчей не считаются подтверждением того, что они были установлены в рабочую копию.


## 0.4.7 — 2026-10-30

- Normalize the model-friendly `work/` prefix as an alias of `/work/` across workspace/task file tools and Python script execution; bare relative paths remain workspace-relative for backward compatibility.
- Return the resolved path namespace from `write_file` so the model can see whether it wrote to `workspace` or the task's `work` directory.
- Allow a successful rewrite/copy of the exact Python script to unlock a retry after a previous `run_python` execution error, not only after a missing-file error.
- Add video inputs to chat uploads and `load_file_for_model`. Since Ollama's chat API accepts images but not native video payloads, videos are sampled into up to 12 ordered, timestamped frames and sent as images. Audio is not analysed.
- Require `ffmpeg`/`ffprobe` for video sampling; support video files up to 100 MB in workspace and 12 MB in direct chat uploads, with a maximum sampled duration of 30 minutes.
- Add regression tests for path aliases, recovery after script rewrites, and bounded video frame sampling.
