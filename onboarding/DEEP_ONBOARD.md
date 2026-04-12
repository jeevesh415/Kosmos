# Kosmos: Agent Onboarding

> Codebase: 802 files, ~2,455,608 tokens
> Generated: 2026-04-02 from commit `3ff33c376c`
> Crawl: 46/46 tasks, 20 modules read
> For complete class skeletons and import graphs: `/tmp/xray/xray.md`

---

## Document Map

| Section | What It Covers |
|---------|---------------|
| Critical Paths | Entry-to-side-effect call chains with every hop cited |
| Module Behavioral Index | Per-module behavioral summary, runtime deps, danger flags |
| Change Impact Index | Hub modules, blast radii, safe vs dangerous changes |
| Key Interfaces | Most important class/function signatures |
| Data Contracts | Cross-boundary Pydantic models, schema evolution risks |
| Error Handling Strategy | Dominant patterns, retries, deviations, exception taxonomy |
| Shared State | Module-level mutables, singletons, caches, thread safety |
| Domain Glossary | Codebase-specific terminology |
| Configuration Surface | Env vars, config files, feature flags |
| Conventions | Implicit rules with evidence counts |
| Gotchas | Counterintuitive behaviors, severity-tagged, clustered by subsystem |
| Hazards | Files that waste context (large, generated, migrations) |
| Extension Points | Common modification tasks with starting files |
| Change Playbooks | Step-by-step modification checklists with validation |
| Reading Order | Suggested file sequence by architectural layer |
| Environment Bootstrap | Setup instructions and minimum env vars |

**Recommended reading order:** Shared State → Critical Path 1 → Gotchas (Critical+High) → remaining as needed.

**Also available in X-Ray output** (`/tmp/xray/xray.json` and `/tmp/xray/xray.md`): full import graph, dependency distances, circular deps, dead code, complexity per function, side effects, git risk, coupling, author expertise, type coverage, env vars, CLI args, Pydantic validators, TODO/FIXME markers, test mapping, investigation targets.

---
## Identity

Kosmos is an AI-powered autonomous scientific research platform that orchestrates multi-agent workflows for hypothesis generation, experiment design, literature analysis, and result validation across scientific domains (biology, chemistry, materials science, neuroscience, physics). Built with Python/FastAPI/asyncio, it uses LLM providers (Anthropic, OpenAI, LiteLLM) via a pluggable provider abstraction, stores research state in PostgreSQL/Neo4j/Redis, and exposes both a CLI (Click/Typer) and WebSocket API for real-time research session streaming.

---
## Critical Paths

### Path 0: CLI Entry Point to Database-Persisted Research Results (Calibration Trace)

```
cli_entrypoint() (kosmos/cli/main.py:422) (see Module Index: cli_main.py)
  -> app() -> Typer callback main() (kosmos/cli/main.py:98)
       -> setup_logging() (kosmos/cli/main.py:49) — configures file handler to get_log_dir()/kosmos.log (see Module Index: core_logging.py)
       |    [FACT] If trace or debug_level >= 2, level=DEBUG; if debug or debug_level >= 1, level=DEBUG;
       |    if verbose, level=INFO; else WARNING (main.py:64-70).
       |    [FACT] When trace=True, mutates global config object: sets log_llm_calls, log_agent_messages,
       |    log_workflow_transitions, stage_tracking_enabled all to True, debug_level to 3 (main.py:76-83).
       -> [SIDE EFFECT: database creation] init_from_config() (kosmos/db/__init__.py:140) (see Module Index: db.py)
       |    [FACT] Calls first_time_setup(database_url) which creates .env if missing and runs migrations
       |    (db/__init__.py:170). Then calls init_database() which creates SQLAlchemy engine, session factory,
       |    and calls Base.metadata.create_all() (db/__init__.py:99-100). SQLite gets check_same_thread=False
       |    (db/__init__.py:74); PostgreSQL gets QueuePool with pool_pre_ping=True (db/__init__.py:87).
       |  x on failure: database init failure is caught but NOT fatal (main.py:148-165) (see Gotcha #107).
       |    [FACT] Comment: "Don't exit here - let commands handle the error if they need database"
       |    (main.py:164). User sees Rich error panel unless quiet=True.
       -> register_commands() (kosmos/cli/main.py:397) — imports and registers 7 command modules
       |    [FACT] Called at module import time, not inside callback (main.py:419). Imports: run, status,
       |    history, cache, config, profile, graph (main.py:401). Registers each with app.command()
       |    (main.py:403-410).
       |    [FACT] ImportError silently swallowed with logging.debug (main.py:413-414) -- missing commands
       |    produce no user-visible error.
       -> run_research() (kosmos/cli/commands/run.py:51)
            [FACT] Typer command accepting: question (positional, optional), domain, max_iterations
            (default 10, run.py:54), budget, data_path, no_cache, interactive, output, stream,
            stream_tokens (run.py:51-61).
            |
            | Branching at entry:
            |  - If interactive=True OR question is None -> run_interactive_mode() (run.py:87-88).
            |    Returns config dict or None. If None, exits 0 (run.py:91-92).
            |  - If question still None after interactive -> print_error() + exit 1 (run.py:102-104)
            |  - If data_path provided but not exists -> print_error() + exit 1 (run.py:107-109)
            |
            | Data transformation: CLI params -> flat_config dict (run.py:148-170).
            | [FACT] Nested KosmosConfig structure is flattened because "Agents expect flat keys, not
            | nested KosmosConfig structure" (run.py:147). Fields include max_iterations,
            | enabled_domains, budget_usd, enable_concurrent_operations, llm_provider, data_path
            | (run.py:150-170).
            |
            -> ResearchDirectorAgent(research_question, domain, config=flat_config) (run.py:173-177) (see Module Index: research_director.py)
            |    [FACT] Inherits from BaseAgent (base.py:97) (see Module Index: kosmos/agents/base.py). BaseAgent assigns UUID agent_id,
            |    status=CREATED, initializes message queues, statistics counters (base.py:127-153).
            |    -> _validate_domain() (research_director.py:94) — warns if domain not in enabled_domains
            |    |    default list: ["biology", "physics", "chemistry", "neuroscience"]
            |    |    (research_director.py:265). [FACT] Does NOT reject unknown domains, only logs
            |    |    warning (research_director.py:270-272).
            |    -> _load_skills() (research_director.py:98) — SkillLoader().load_skills_for_task().
            |    |    Failure caught, sets self.skills = None (research_director.py:306).
            |    -> get_client() (kosmos/core/llm.py:613) — thread-safe singleton LLM client (see Module Index: core_llm.py)
            |    |    [FACT] Thread-safe singleton with _client_lock (llm.py:646). Uses double-checked
            |    |    locking (llm.py:648-649). Tries get_provider_from_config() first; on failure
            |    |    falls back to AnthropicProvider with env var API key (llm.py:654-673).
            |    -> ResearchPlan(research_question, domain, max_iterations) (research_director.py:116-120)
            |    |    — Pydantic model, tracks hypothesis_pool, experiment_queue, completed_experiments
            |    |    as string ID lists (workflow.py:57-78) (see Module Index: core_workflow.py; see Data Contracts).
            |    -> ResearchWorkflow(initial_state=INITIALIZING) (research_director.py:122-125)
            |    |    — state machine with explicit ALLOWED_TRANSITIONS dict (workflow.py:175-227) (see Module Index: core_workflow.py).
            |    |    [FACT] 9 states: INITIALIZING, GENERATING_HYPOTHESES, DESIGNING_EXPERIMENTS,
            |    |    EXECUTING, ANALYZING, REFINING, CONVERGED, PAUSED, ERROR (workflow.py:19-29).
            |    -> ConvergenceDetector(mandatory_criteria, optional_criteria, config)
            |    |    (research_director.py:168-178) — initialized with max_iterations,
            |    |    novelty_decline_threshold=0.3, min_experiments_before_convergence=2
            |    |    (research_director.py:174-176).
            |    -> get_world_model() (research_director.py:243) — persists Entity.from_research_question()
            |         to knowledge graph. Failure caught, sets self.wm = None (research_director.py:253)
            |         (see Module Index: world_model_simple.py; see Module Index: knowledge_graph.py).
            |
            |    [PATTERN] Lazy-init agent slots (_hypothesis_agent, _experiment_designer,
            |    _code_generator, _code_executor, _data_provider, _data_analyst, _hypothesis_refiner)
            |    are all None at init (research_director.py:145-151). They are only created on first
            |    use inside action handlers. This means first execution of each phase has extra latency
            |    (see Gotcha #16).
            |
            |    [PATTERN] Both asyncio.Lock AND threading.RLock are maintained in parallel
            |    (research_director.py:193-200). Comment says "Keep threading locks for backwards
            |    compatibility in sync contexts" (research_director.py:197) (see Gotcha #2). The sync locks
            |    (_research_plan_lock_sync, _workflow_lock_sync) are used by _research_plan_context()
            |    and _workflow_context() context managers (research_director.py:363-379), while async
            |    code should use self._research_plan_lock directly.
            |
            |    [PATTERN] init_from_config() is called AGAIN inside __init__ (research_director.py:131-138),
            |    despite already being called in the CLI callback. RuntimeError with "already initialized"
            |    is silently caught (research_director.py:134-136) (see Gotcha #15).
            |
            -> get_registry().register(director) (run.py:181-182) — [SIDE EFFECT: global registry mutation]
            -> asyncio.run(run_with_progress_async(...)) (run.py:186-192) — async bridge
                 -> run_with_progress_async() (kosmos/cli/commands/run.py:225)
                      [FACT] Creates Rich Progress with 5 task bars: hypothesis, experiment, execution,
                      analysis, iteration (run.py:260-274). Wraps execution in Live context (run.py:293).
                      -> await director.execute({"action": "start_research"}) (run.py:296)
                           -> ResearchDirectorAgent.execute() (research_director.py:2868)
                                [FACT] Async method. Routes on task["action"] (research_director.py:2878).
                                |
                                | Branch A: action == "start_research" (research_director.py:2880):
                                |  -> generate_research_plan() (research_director.py:2882) — LLM call #1
                                |  |    [FACT] Sends prompt to self.llm_client.generate(prompt, max_tokens=1000)
                                |  |    (research_director.py:2372). [SIDE EFFECT: LLM API call]. Stores
                                |  |    response in self.research_plan.initial_strategy (research_director.py:2375).
                                |  |    x on failure: returns error string, does NOT raise
                                |  |    (research_director.py:2380-2382).
                                |  -> self.start() (research_director.py:2885) — BaseAgent lifecycle
                                |  |    [FACT] start() (base.py:159) checks status != CREATED, sets status
                                |  |    to STARTING, calls _on_start(), then sets RUNNING (base.py:165-171).
                                |  |    [FACT] _on_start() records _start_time for runtime tracking
                                |  |    (research_director.py:325), then transitions workflow
                                |  |    INITIALIZING -> GENERATING_HYPOTHESES (research_director.py:328-332).
                                |  |    Uses sync _workflow_lock_sync (research_director.py:328).
                                |  -> decide_next_action() (research_director.py:2388)
                                |  -> await _execute_next_action(next_action) (research_director.py:2889)
                                |
                                | Branch B: action == "step" (research_director.py:2897):
                                |  -> decide_next_action() (research_director.py:2899)
                                |  -> await _execute_next_action(next_action) (research_director.py:2900)

  decide_next_action() (research_director.py:2388) — pre-checks and state-based routing
    [FACT] Pre-checks before state-based routing:
    1. Budget enforcement via get_metrics().enforce_budget() (research_director.py:2406-2409).
       BudgetExceededError -> CONVERGE (research_director.py:2410-2422).
       ImportError silently caught (research_director.py:2423-2425).
    2. Runtime limit check: _check_runtime_exceeded() (research_director.py:2428). Compares
       elapsed hours against max_runtime_hours (default 12h, research_director.py:105).
    3. Infinite loop guard: _actions_this_iteration counter, MAX_ACTIONS_PER_ITERATION=50
       (research_director.py:50). If exceeded, forces CONVERGE (research_director.py:2455-2461).

    State-based decision tree (research_director.py:2483-2548):
    - GENERATING_HYPOTHESES -> GENERATE_HYPOTHESIS (research_director.py:2483-2485)
    - DESIGNING_EXPERIMENTS -> DESIGN_EXPERIMENT if untested, else EXECUTE_EXPERIMENT if queue,
      else ANALYZE_RESULT if results, else CONVERGE (research_director.py:2487-2505)
    - EXECUTING -> EXECUTE_EXPERIMENT if queue, else ANALYZE_RESULT if results,
      else REFINE_HYPOTHESIS (research_director.py:2507-2519)
    - ANALYZING -> ANALYZE_RESULT if results, else fallback to EXECUTE or REFINE
      (research_director.py:2521-2531)
    - REFINING -> REFINE_HYPOTHESIS if tested, else GENERATE_HYPOTHESIS
      (research_director.py:2533-2538)
    - CONVERGED -> CONVERGE (research_director.py:2540-2541)
    - ERROR -> ERROR_RECOVERY (research_director.py:2543-2544)

  _execute_next_action() -> _do_execute_action() (research_director.py:2550, 2573)
    [FACT] Wrapped by get_stage_tracker().track() context manager for telemetry
    (research_director.py:2570).

    Six action branches in _do_execute_action():

    GENERATE_HYPOTHESIS -> _handle_generate_hypothesis_action() (research_director.py:1391):
      [FACT] Lazy-inits HypothesisGeneratorAgent(config=self.config) (research_director.py:1403-1404).
      Calls generate_hypotheses(research_question, num_hypotheses=3, domain, store_in_db=True)
      (research_director.py:1408-1413). On success: adds IDs to research_plan, persists to knowledge
      graph, transitions to DESIGNING_EXPERIMENTS (research_director.py:1427-1447).
      x on failure: _handle_error_with_recovery() (research_director.py:1449-1456).

    DESIGN_EXPERIMENT -> _handle_design_experiment_action(hypothesis_id) (research_director.py:1458):
      [FACT] Lazy-inits ExperimentDesignerAgent(config=self.config) (research_director.py:1469-1470).
      Calls design_experiment(hypothesis_id, store_in_db=True) (research_director.py:1474-1477).
      On success: adds protocol_id to experiment_queue, persists to graph, transitions to EXECUTING
      (research_director.py:1490-1510). Concurrent path: if enable_concurrent and async_llm_client
      and multiple untested, evaluates batch concurrently via evaluate_hypotheses_concurrently()
      with 300s timeout (research_director.py:2585-2616).
      x on failure: _handle_error_with_recovery()

    EXECUTE_EXPERIMENT -> _handle_execute_experiment_action(protocol_id) (research_director.py:1521):
      [FACT] Lazy-inits ExperimentCodeGenerator(use_templates=True, use_llm=True)
      (research_director.py:1538), CodeExecutor(max_retries=3) (research_director.py:1540),
      DataProvider() (research_director.py:1541-1543).
      Sub-hops:
        1. Loads protocol from DB via get_experiment(session, protocol_id)
           (research_director.py:1552). Reconstructs ExperimentProtocol.model_validate(protocol_data)
           from stored JSON (research_director.py:1559).
        2. self._code_generator.generate(protocol) (research_director.py:1564) — template matching
           first (5 templates registered: TTest, Correlation, LogLogScaling, ML,
           GenericComputational; code_generator.py:787-792), then LLM fallback, then basic template
           (code_generator.py:807-833). [SIDE EFFECT: potential LLM API call for code generation]
        3. self._code_executor.execute(code, retry_on_error=True) (research_director.py:1572)
           [SIDE EFFECT: arbitrary Python code execution].
           [FACT] Uses restricted builtins (SAFE_BUILTINS dict at executor.py:43-83) with
           _make_restricted_import() that whitelists scientific modules only (executor.py:86-94,
           allowed: numpy, pandas, scipy, sklearn, matplotlib, etc.).
           Timeout: 300s default (executor.py:40). Retry loop up to max_retries=3 with
           RetryStrategy.modify_code_for_retry() for self-correcting execution
           (executor.py:316-324).
        4. Extracts p_value, effect_size, statistical metrics from exec_result.return_value
           (research_director.py:1578-1591).
        5. JSON-sanitizes via _json_safe() (research_director.py:1595-1613) — converts numpy
           types to Python primitives, non-serializable objects to strings.
        6. [SIDE EFFECT: database INSERT] create_result(session, ...) (research_director.py:1621-1630).
           Stores result_id (UUID), experiment_id, data, p_value, effect_size, statistical_tests.
        7. Updates research_plan: add_result(), mark_experiment_complete()
           (research_director.py:1641-1642).
        8. Persists to knowledge graph (research_director.py:1645-1648).
        9. Transitions to ANALYZING (research_director.py:1651-1655).
      x on failure: _handle_error_with_recovery()

    ANALYZE_RESULT -> _handle_analyze_result_action(result_id) (research_director.py:1666):
      [FACT] Lazy-inits DataAnalystAgent(config=self.config) (research_director.py:1680-1681).
      Loads result + experiment + hypothesis from DB, reconstructs Pydantic models
      (research_director.py:1690-1734). Calls _data_analyst.interpret_results(result, hypothesis)
      (research_director.py:1737-1740). [SIDE EFFECT: LLM API call for interpretation].
      Updates hypothesis status: mark_supported or mark_rejected based on
      interpretation.hypothesis_supported (research_director.py:1762-1767). Adds
      SUPPORTS/REFUTES relationship to knowledge graph (research_director.py:1770-1778).
      Transitions to REFINING (research_director.py:1781-1785).
      x on failure: _handle_error_with_recovery()

    REFINE_HYPOTHESIS -> _handle_refine_hypothesis_action(hypothesis_id) (research_director.py:1796):
      [FACT] Lazy-inits HypothesisRefiner(config=self.config) (research_director.py:1816-1817).
      Loads hypothesis and all its experiment results from DB (research_director.py:1825-1872).
      [FACT] If no results found, skips refinement and returns early
      (research_director.py:1877-1883). After refinement: increments iteration counter and resets
      _actions_this_iteration = 0 (research_director.py:2703-2706).
      x on failure: _handle_error_with_recovery()

    CONVERGE -> _handle_convergence_action() (research_director.py:1334):
      [FACT] First applies Benjamini-Hochberg FDR multiple comparison correction to p-values
      (research_director.py:1343-1345). Then calls _check_convergence_direct()
      (research_director.py:1347) which invokes
      self.convergence_detector.check_convergence(research_plan, hypotheses, results, total_cost)
      (research_director.py:1267-1271).
      If decision.should_stop: sets research_plan.has_converged = True
      (research_director.py:1357), adds convergence annotation to knowledge graph
      (research_director.py:1361-1371), transitions to CONVERGED state
      (research_director.py:1374-1378), calls self.stop() (research_director.py:1381).
      If not converged: increments iteration and resets action counter
      (research_director.py:1385-1389).

    ERROR_RECOVERY (research_director.py:2716-2728):
      [FACT] Resets _consecutive_errors = 0 (research_director.py:2718). Transitions
      ERROR -> GENERATING_HYPOTHESES (research_director.py:2723-2727). This is the only
      recovery path from ERROR state.

  Research Loop (kosmos/cli/commands/run.py:308-391):
    [FACT] Main loop: while iteration < max_iterations (run.py:308). 2-hour hard timeout
    (run.py:301). Each iteration: calls director.get_research_status() (run.py:326), checks
    has_converged (run.py:362-365), then await director.execute({"action": "step"}) (run.py:369).
    50ms async sleep between iterations for UI updates (run.py:389).

    Terminal side effects at loop end:
    -> Fetches hypothesis and experiment objects from DB using IDs in research_plan
       (run.py:404-435).
    -> Builds results dict with metrics including api_calls, cache_hits, cache_misses,
       hypotheses_generated/tested/supported/rejected, experiments_executed (run.py:437-458).

  Result Display (kosmos/cli/commands/run.py:195-212):
    -> ResultsViewer.display_research_overview(results) (run.py:196) — Rich Panel output
       (results_viewer.py:49-80)
    -> ResultsViewer.display_hypotheses_table(results["hypotheses"]) (run.py:197)
    -> ResultsViewer.display_experiments_table(results["experiments"]) (run.py:198)
    -> ResultsViewer.display_metrics_summary(results["metrics"]) (run.py:200-201)
    -> Optional: viewer.export_to_json(results, output) or viewer.export_to_markdown(results, output)
       (run.py:204-210) — [SIDE EFFECT: file write]
```

**Error Recovery Architecture:** (see Error Handling Strategy)

[FACT] `_handle_error_with_recovery()` (research_director.py:599) implements three-tier error handling:
1. Consecutive error counter (`_consecutive_errors`) with MAX_CONSECUTIVE_ERRORS=3 circuit breaker (research_director.py:45-46).
2. Exponential backoff: [2, 4, 8] seconds (research_director.py:46).
3. On circuit break: transitions to ERROR state (research_director.py:649-662). On recoverable error below threshold: backs off and suggests retry via returned NextAction (research_director.py:665-668).

[FACT] Every action handler calls `_reset_error_streak()` on success (e.g., research_director.py:1424, 1488, 1637, 1757). This means a single success resets the circuit breaker counter.

**LLM Call Sites (Complete Enumeration):**

1. `generate_research_plan()` -- `llm_client.generate(prompt, max_tokens=1000)` (research_director.py:2372)
2. `HypothesisGeneratorAgent._detect_domain()` -- `llm_client.generate(prompt, max_tokens=50, temperature=0.0)` (hypothesis_generator.py:277-280)
3. `HypothesisGeneratorAgent._generate_with_claude()` -- `llm_client.generate_structured(prompt, schema, max_tokens=4000, temperature=0.7)` (hypothesis_generator.py:378-383)
4. `ExperimentCodeGenerator._generate_with_llm()` -- `llm_client.generate(prompt)` (code_generator.py:847) -- only when no template matches
5. `DataAnalystAgent.interpret_results()` -- LLM call for result interpretation
6. `HypothesisRefiner` -- LLM call for hypothesis refinement

[FACT] `ClaudeClient.generate()` (llm.py:207) checks cache before API call. Cache key includes prompt, model, system, max_tokens, temperature, stop_sequences (llm.py:278-289). Cache hit increments `cache_hits` counter and returns immediately (llm.py:293-299). Cache miss calls `self.client.messages.create()` (llm.py:309-316), extracts `response.content[0].text` (llm.py:333), caches response (llm.py:336-358).

[FACT] Auto model selection: if `enable_auto_model_selection=True` and not CLI mode, `ModelComplexity.estimate_complexity()` scores prompt 0-100 (llm.py:53-105). Score < 30 -> haiku; >= 30 -> sonnet (llm.py:88-93). Keywords like "research", "hypothesis", "experiment" add 10 points each (llm.py:45-50) (see Module Index: core_llm.py; see Configuration Surface).

**Terminal Side Effects (Complete):**

1. **Database**: Table creation at startup (db/__init__.py:100). Hypothesis INSERT (hypothesis_generator.py:474-490). Experiment result INSERT (research_director.py:1621-1630). Session auto-commits on context exit (db/__init__.py:133).
2. **LLM API**: Anthropic `messages.create()` calls (llm.py:309-316). Token counters updated per call (llm.py:320-324).
3. **Knowledge Graph**: Entity and Relationship creation via `WorldModel` (research_director.py:243-255, 388-436, 438-475, 477-524, 526-562).
4. **File System**: Log file writes to `get_log_dir()/kosmos.log` (main.py:61). Optional result export to JSON/Markdown (run.py:204-210).
5. **Console**: Rich live progress display (run.py:293). Result tables and panels (run.py:195-201).
6. **Code Execution**: `exec()` of generated Python in restricted sandbox (executor.py:617). Optional Docker sandbox (executor.py:576).

---

### Path 1: P1.2: FastAPI Health Endpoint Trace

```
get_health_checker() (kosmos/api/health.py:401-411) — module-level singleton
  [FACT] health.py:398 declares _health_checker: Optional[HealthChecker] = None.
  HealthChecker.__init__() (line 31) records self.start_time = time.time() for uptime calculation.
  This means uptime is measured from first health check invocation, not from process start.

  [ABSENCE] grep-confirmed no FastAPI() instantiation or include_router calls referencing health
  in kosmos/. No dedicated FastAPI app mounting these as HTTP routes.

  Three consumption paths:
  1. Direct import by monitoring/alerting subsystem (kosmos/monitoring/alerts.py)
  2. Re-export through kosmos/api/__init__.py (lines 7-13) for programmatic use
  3. kosmos/api/streaming.py:169-181 defines a /stream/health endpoint on the streaming router,
     but this is a *separate* streaming-specific health check, not the main one.

  The CLI doctor command (kosmos/cli/main.py:267-392) (see Module Index: cli_main.py) implements its own parallel health check
  logic rather than delegating to HealthChecker (see Gotcha #109) -- it directly calls get_session() and
  validate_database_schema() [FACT: kosmos/cli/main.py:321-326].

  -> get_basic_health() (health.py:415-417 -> health.py:36-55) — liveness probe
  |    Increments checks_performed, records timestamp, computes uptime from self.start_time,
  |    and calls _get_version().
  |    -> _get_version() (health.py:369-375) — imports kosmos.__version__ which is "0.2.0"
  |         [FACT: kosmos/__init__.py:12]. Catches all exceptions and returns "unknown" on
  |         failure -- safe but silent.
  |    No external I/O. This is a pure in-process check. Response time is effectively zero.

  -> get_readiness_check() (health.py:57-106) — readiness probe
  |    Calls four sub-checks sequentially and aggregates results.
  |    [FACT] Critical design decision: Neo4j failure does NOT affect overall readiness
  |    (health.py:96-97, comment "Don't mark as not ready if Neo4j is down (it's optional)").
  |    Only database, cache, and external API failures set all_ready = False.
  |    Return structure: {"status": "ready"|"not_ready", "components": {...}, ...}
  |
  |    -> _check_database() (health.py:187-215)
  |    |    -> from kosmos.db import get_session — lazy import at call time [FACT: health.py:195]
  |    |    -> get_session() is a @contextmanager at kosmos/db/__init__.py:108-137
  |    |    |    Pre-condition: _SessionLocal must not be None, otherwise raises
  |    |    |    RuntimeError("Database not initialized. Call init_database() first.")
  |    |    |    [FACT: db/__init__.py:126-127]
  |    |    -> session.execute("SELECT 1") [FACT: health.py:200] and measures round-trip time
  |    |    On success: returns {"status": "healthy", "response_time_ms": ...}
  |    |    x on failure: catches broadly, logs error, returns {"status": "unhealthy", "error": str(e)}
  |    |
  |    |    [PATTERN] The get_session() context manager calls session.commit() on normal exit
  |    |    (db/__init__.py:133). A bare SELECT 1 produces a commit on a read-only query, which
  |    |    is harmless but unnecessary. On exception, it calls session.rollback() then re-raises
  |    |    -- but the health check's outer try/except catches that re-raise.
  |    |
  |    |    Database engine config: init_database() at db/__init__.py:26-105 configures:
  |    |    - SQLite: no connection pooling, check_same_thread=False
  |    |    - PostgreSQL/others: QueuePool with pool_pre_ping=True (stale connection detection),
  |    |      default pool_size=5, max_overflow=10
  |    |    - Slow query logging via SQLAlchemy event hooks (db/operations.py:51-80) if enabled
  |    |
  |    -> _check_cache() (health.py:217-272)
  |    |    -> os.getenv("REDIS_ENABLED") (default "false") [FACT: health.py:226]
  |    |    |
  |    |    | If Redis disabled: immediately returns {"status": "healthy", "type": "memory"}
  |    |    | -- in-memory cache is assumed always healthy
  |    |    |
  |    |    | If Redis enabled: import redis (lazy), creates client from REDIS_URL with 5s
  |    |    | socket timeouts [FACT: health.py:234-238], calls client.ping() then client.info()
  |    |    | On success: returns version, used_memory_mb, connected_clients from Redis INFO
  |    |    |
  |    |    | [PATTERN] BUG: redis_enabled variable scoping in except block. At line 269, the except
  |    |    | handler references redis_enabled in the expression "redis" if redis_enabled else "memory".
  |    |    | If the exception occurs *before* line 226 executes (impossible in current code since 226
  |    |    | is the first line of the try), redis_enabled would be unbound. Latent UnboundLocalError
  |    |    | risk, though practically unreachable.
  |    |    |
  |    |    | [PATTERN] Config bypass: The health check reads REDIS_ENABLED and REDIS_URL directly
  |    |    | from os.getenv() rather than from the RedisConfig Pydantic model (config.py:305-316)
  |    |    | (see Gotcha #55; see Configuration Surface).
  |    |    | Config validation/defaults in the Pydantic model are bypassed. The defaults happen to
  |    |    | match ("false" and "redis://localhost:6379/0"), but any config override that only sets
  |    |    | the Pydantic model without setting env vars will be invisible to the health check.
  |    |
  |    -> _check_external_apis() (health.py:274-324)
  |    |    -> os.getenv("ANTHROPIC_API_KEY") -> format validation only
  |    |    [FACT] No network call. This check explicitly avoids making API calls
  |    |    (health.py:302, comment "we don't want to make actual API calls in health checks").
  |    |    Validates:
  |    |    1. Key exists in env
  |    |    2. If key is all-9s: CLI mode (healthy)
  |    |    3. If key starts with sk-ant-: API mode (healthy)
  |    |    4. Otherwise: warning status
  |    |
  |    |    Three-state return: "healthy", "unhealthy" (no key), or "warning" (unexpected format).
  |    |    [FACT] The warning state DOES block readiness. The readiness check at line 90 is:
  |    |    if api_status["status"] != "healthy". A "warning" status is not "healthy", so it
  |    |    WILL set all_ready = False. This means an API key with unexpected format (e.g., an
  |    |    OpenAI key in ANTHROPIC_API_KEY) will mark the entire system as not ready.
  |    |
  |    -> _check_neo4j() (health.py:326-367)
  |         -> from neo4j import GraphDatabase — lazy import [FACT: health.py:334]
  |         -> Reads NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD from env vars
  |         |  If no password: returns {"status": "not_configured"} (skips connectivity test)
  |         -> Creates driver, calls driver.verify_connectivity() (Neo4j bolt protocol handshake)
  |         -> Closes driver after check (no connection reuse)
  |         x on failure: logs at WARNING level (not ERROR), returns {"status": "unavailable"}
  |
  |         [FACT] Degraded mode: Neo4j is explicitly optional. Its failure uses "unavailable"
  |         status (not "unhealthy"), and the readiness check skips it for the all_ready flag.
  |         The Neo4jConfig in config.py:534-552 has default password "kosmos-password", but
  |         the health check reads from env vars directly, so the Pydantic default is invisible.
  |
  |         [PATTERN] Resource leak potential: If driver.verify_connectivity() raises,
  |         driver.close() is never called (no try/finally around it). The outer except catches
  |         the error, but the driver may hold open sockets until GC.

  -> get_metrics() (health.py:108-185) — system metrics probe
       -> psutil.cpu_percent(interval=0.1) — blocks for 100ms [FACT: health.py:119]
       -> psutil.virtual_memory(), psutil.disk_usage('/'), psutil.Process() for process stats
       -> _get_load_average() (health.py:377-385): Unix-only via os.getloadavg(), returns None on Windows
       -> _get_num_fds() (health.py:387-394): Unix-only via process.num_fds(), returns None on Windows
       [FACT] Always returns {"status": "healthy"} regardless of resource pressure -- no
       threshold-based degradation.
```

**Cross-System Consumers:**

The `AlertManager` class reuses health checks for its alert rules:
- `_check_database_connection()` (alerts.py:273-282): imports `get_health_checker()`, calls `_check_database()`, returns True if unhealthy. Registered as CRITICAL severity with 60s cooldown [FACT: alerts.py:141-147].
- `_check_cache_availability()` (alerts.py:324-333): same pattern, calls `_check_cache()`, returns True if unhealthy. Registered as WARNING severity with 120s cooldown [FACT: alerts.py:195-201].

[PATTERN] This creates a shared singleton: both the alerting system and any direct health endpoint callers use the same `_health_checker` instance, sharing `checks_performed` and `last_check_time` counters.

The CLI `doctor` command (kosmos/cli/main.py:267-392) does NOT use `HealthChecker`. It independently:
- Calls `get_session()` with an empty body (just tests connection) [FACT: main.py:326-327]
- Runs `validate_database_schema()` for schema completeness
- Checks package imports and API keys
- Does not check Redis, Neo4j, or system resources

**Summary Call Chain:**

```
get_readiness_check() [health.py:57]
  +-- _check_database() [health.py:187]
  |     +-- kosmos.db.get_session() [db/__init__.py:109]
  |           +-- _SessionLocal() -> session.execute("SELECT 1") -> session.commit()
  |                 +-- Pre-req: init_database() must have been called
  +-- _check_cache() [health.py:217]
  |     +-- [REDIS_ENABLED=false] -> immediate "healthy" (in-memory)
  |     +-- [REDIS_ENABLED=true] -> redis.from_url() -> .ping() -> .info()
  +-- _check_external_apis() [health.py:274]
  |     +-- os.getenv("ANTHROPIC_API_KEY") -> format check only (no network)
  +-- _check_neo4j() [health.py:326]
        +-- neo4j.GraphDatabase.driver() -> .verify_connectivity() -> .close()
              +-- Pre-req: NEO4J_PASSWORD must be set (else "not_configured")
```

---

### Path 2: P1.3: Experiment Execution Trace

```
ExperimentDesignerAgent.execute() (experiment_designer.py:109)
  [FACT] When task_type == "design_experiment" (experiment_designer.py:124), calls
  design_experiment() at line 131.
  x on failure: top-level execute() catches all exceptions, sets status to ERROR, and
  returns an AgentMessage with MessageType.ERROR [FACT] (experiment_designer.py:150-159).
  Status resets to IDLE in the finally block.

  -> design_experiment() (experiment_designer.py:164) — 8-step pipeline
  |
  |  Step 1: Load hypothesis from DB if only ID provided [FACT] (experiment_designer.py:195-196).
  |           Uses SQLAlchemy get_session() with DBHypothesis.
  |
  |  Step 2: Select experiment type via _select_experiment_type() [FACT]
  |           (experiment_designer.py:201). Falls back through: preferred type, hypothesis
  |           suggestion, domain heuristic map (line 394-406), then default COMPUTATIONAL.
  |
  |  Step 3: Generate protocol via template or LLM
  |    -> _generate_from_template() (experiment_designer.py:409) — uses
  |    |    TemplateRegistry.find_best_template() [FACT] (experiment_designer.py:418).
  |    |    If no template found, falls back to LLM generation [FACT]
  |    |    (experiment_designer.py:421-424).
  |    -> _generate_with_claude() (experiment_designer.py:441) — LLM path calls
  |         llm_client.generate_structured() with a JSON schema and 8192 max tokens
  |         [FACT] (experiment_designer.py:495-499). [SIDE EFFECT: LLM API call]
  |
  |  Step 4: Enhance with LLM if both use_llm_enhancement and use_templates are true
  |           [FACT] (experiment_designer.py:221).
  |
  |  Step 5: Validate protocol via ExperimentValidator from kosmos.experiments.validator,
  |           with inline fallback validation if the validator fails [FACT]
  |           (experiment_designer.py:720-739). Inline validation checks control groups,
  |           sample size, variables, and statistical tests.
  |
  |  Step 6: Calculate metrics:
  |           rigor score (0.0-1.0) [FACT] (experiment_designer.py:779),
  |           completeness score [FACT] (experiment_designer.py:807),
  |           feasibility (High/Medium/Low) [FACT] (experiment_designer.py:836).
  |
  |  Step 7: [SIDE EFFECT: database INSERT] Store in database via _store_protocol() [FACT]
  |           (experiment_designer.py:271-272). Creates DBExperiment with status CREATED.
  |           Error handling tolerates missing DB and schema mismatches without crashing
  |           [FACT] (experiment_designer.py:910-918).
  |
  |  Step 8: Return ExperimentDesignResponse with protocol, scores, and recommendations
  |           [FACT] (experiment_designer.py:277-292).

  -> execute_protocol_code() (executor.py:1017) — code execution pipeline
       [FACT] The validate_safety bypass was explicitly removed per comment
       "F-21: removed validate_safety bypass" (executor.py:1027).

       -> CodeValidator.validate() (code_validator.py:159) — 6 sequential safety checks (see Module Index: guardrails.py; see Gotcha #28)
       |    1. _check_syntax() via ast.parse() [FACT] (code_validator.py:233-244).
       |    2. _check_dangerous_imports() via AST walking -- checks Import and ImportFrom
       |    |    nodes against a blocklist of 14 modules including os, subprocess, sys, shutil,
       |    |    pickle [FACT] (code_validator.py:35-39). Falls back to string matching if AST
       |    |    parse fails [FACT] (code_validator.py:272-280).
       |    3. _check_dangerous_patterns() -- string search for eval(, exec(, compile(,
       |    |    __import__, globals(, etc. [FACT] (code_validator.py:42-53). Special handling
       |    |    for open(): allowed in read-only mode if allow_file_read=True, blocked in
       |    |    write mode [FACT] (code_validator.py:291-311).
       |    4. _check_ast_calls() -- detects getattr(), setattr(), delattr() and dunder
       |    |    attribute access (__dict__, __class__, __builtins__, __subclasses__)
       |    |    [FACT] (code_validator.py:322-374).
       |    5. Network operations -- warning-level keyword search for socket, http, urllib,
       |    |    etc. [FACT] (code_validator.py:377-384).
       |    6. _check_ethical_guidelines() -- keyword-based check against configurable
       |         guidelines covering harm, PII, human subjects, animal welfare, and
       |         environmental impact [FACT] (code_validator.py:386-418). Five default
       |         guidelines loaded if no custom file provided [FACT] (code_validator.py:109-157).
       |
       |    [FACT] Risk assessment: Overall risk = max severity across all violations
       |    (code_validator.py:421-434). If risk_level is HIGH or CRITICAL, the
       |    requires_approval() method returns True, gating on human review
       |    [FACT] (code_validator.py:436-465).
       |
       |    x If validation fails (report.passed == False): execute_protocol_code returns early
       |    with success: False and validation_errors list [FACT] (executor.py:1042-1048).
       |    Code never reaches execution.

       -> CodeExecutor.execute() (executor.py:237) — two execution modes (see Module Index: executor.py)
            |
            | Path A -- Docker Sandbox (default, use_sandbox=True):
            |    Constructor attempts Docker initialization; falls back gracefully if Docker
            |    unavailable [FACT] (executor.py:216-224). Sets self.use_sandbox = False on failure
            |    (see Gotcha #22).
            |    -> _execute_in_sandbox() (line 555) — prepares data files
            |    |    -> DockerSandbox.execute() (sandbox.py:163)
            |    |         -> Creates temp directory, writes code to experiment.py, copies data
            |    |         |    files [FACT] (sandbox.py:181-208). Temp directory cleaned up in
            |    |         |    finally block [FACT] (sandbox.py:212-215).
            |    |         -> _run_container() (sandbox.py:217) — strict security:
            |    |              mem_limit and nano_cpus enforce resource bounds
            |    |              [FACT] (sandbox.py:265-266).
            |    |              network_disabled=True blocks all network access
            |    |              [FACT] (sandbox.py:267).
            |    |              read_only=True filesystem with tmpfs exceptions for /tmp (100MB)
            |    |              and /home/sandbox/.local (50MB) [FACT] (sandbox.py:268-272).
            |    |              security_opt=['no-new-privileges'] and cap_drop=['ALL']
            |    |              [FACT] (sandbox.py:273-274).
            |    |              Code directory mounted read-only, output directory mounted
            |    |              read-write [FACT] (sandbox.py:241-242).
            |    |              Monitoring: daemon thread tracks CPU and memory usage via
            |    |              container.stats() stream [FACT] (sandbox.py:289-296, 402-436).
            |    |              Timeout: container.wait(timeout=self.timeout) with graceful stop
            |    |              (5s) then kill fallback [FACT] (sandbox.py:304-318). Default
            |    |              timeout is 300 seconds [FACT] (sandbox.py:84).
            |    |              Return value extraction: scans stdout for RESULT: prefix lines,
            |    |              parses as JSON [FACT] (sandbox.py:438-450).
            |    |              [SIDE EFFECT: Docker container lifecycle]
            |    |              Container cleanup: container.remove(force=True) in finally block
            |    |              [FACT] (sandbox.py:396-400).
            |
            | Path B -- Restricted Direct Execution (fallback):
            |    -> _prepare_globals() (line 589) — replaces __builtins__ with SAFE_BUILTINS,
            |    |    a curated whitelist of 80+ safe builtins [FACT] (executor.py:43-83).
            |    |    Dangerous builtins like open, __import__ (unrestricted), eval, exec excluded.
            |    |    A restricted __import__ function only permits 30+ scientific modules:
            |    |    numpy, pandas, scipy, sklearn, matplotlib, etc. [FACT] (executor.py:86-94).
            |    |    Attempts to import anything else raise ImportError with the allowed list.
            |    -> _exec_with_timeout() (line 600) — uses signal.SIGALRM on Unix or
            |         ThreadPoolExecutor on Windows [FACT] (executor.py:607-630). Default timeout:
            |         300 seconds [FACT] (executor.py:41).
            |    [SIDE EFFECT: exec() of generated Python in restricted namespace]
            |    Return value extraction: looks for results or result variable in exec_locals
            |    [FACT] (executor.py:516).
            |
            | [PATTERN] 2/2 execution paths (sandbox + direct) have timeout protection.
            | [PATTERN] 1/2 paths have full OS-level isolation (Docker only).

       -> RetryStrategy (executor.py:667) — self-correcting retry
            [FACT] Implements Issue #54's paper claim: "reads traceback -> fixes code ->
            re-executes" (executor.py:677).
            Retry loop in CodeExecutor.execute() (line 237) runs up to max_retries attempts:
            1. On failure, modify_code_for_retry() (line 751) tries LLM-based repair first
               (attempts 1-2 only) [FACT] (executor.py:779-788), then falls back to
               pattern-based fixes covering 11 error types: KeyError, FileNotFoundError,
               NameError, TypeError, IndexError, AttributeError, ValueError,
               ZeroDivisionError, ImportError, PermissionError, MemoryError
               [FACT] (executor.py:791-825).
            2. COMMON_IMPORTS dict provides auto-fix for missing imports (pd, np, plt, sns,
               scipy, etc.) [FACT] (executor.py:680-697).
            3. FileNotFoundError and SyntaxError are non-retryable
               [FACT] (executor.py:724-730).
            4. Exponential backoff: base_delay * 2^(attempt-1) [FACT] (executor.py:733-734).
            5. Repair statistics tracked per error type [FACT] (executor.py:736-749).
```

**Provenance Tracking (provenance.py -> artifacts.py):**

`CodeProvenance` (provenance.py:72) is a dataclass linking findings to exact source code locations. Key fields: `notebook_path`, `cell_index`, `start_line`, `end_line`, `code_snippet` (truncated to 500 chars) [FACT] (provenance.py:105-110). Reproducibility metadata includes `seed`, `git_sha` (auto-populated via subprocess), `model`, `temperature`, and `data_hash` [FACT] (provenance.py:123-128).

Hyperlink generation: `to_hyperlink()` produces `notebook.ipynb#cell=N&line=M` format [FACT] (provenance.py:150-169), compatible with GitHub, VS Code, and JupyterLab.

Factory method: `create_from_execution()` (provenance.py:250) builds provenance from execution context -- notebook path, code, cell index, hypothesis ID, cycle, and task ID.

Notebook integration: `NotebookGenerator.create_notebook()` (notebook_generator.py:200) builds `cell_line_mappings` during notebook creation -- each code cell gets a mapping with `cell_index`, `start_line`, `end_line`, `code_hash` (SHA256 truncated to 16 chars) [FACT] (notebook_generator.py:243-254). These mappings are stored in `NotebookMetadata.cell_line_mappings` [FACT] (notebook_generator.py:66, 296).

Finding attachment: The `Finding` dataclass in `artifacts.py` carries `code_provenance: Optional[Dict]` [FACT] (artifacts.py:79). When present, `ArtifactStateManager` renders provenance as clickable markdown links in cycle summaries: `[filename](path#cell=N&line=M) (lines X-Y)` [FACT] (artifacts.py:475-483).

**Orchestration Gap:**

[PATTERN] 0/4 delegation routes in `DelegationManager._execute_task()` (delegation.py:359-386) attach `code_provenance` to returned findings. The `_execute_data_analysis` method (delegation.py:388-422) returns a finding dict without `code_provenance` or `notebook_path` keys [FACT] (delegation.py:412-422). Similarly, `_execute_literature_review` and `_execute_hypothesis_generation` omit provenance fields.

[ABSENCE] grep-confirmed: `grep -n "provenance|notebook_gen|CodeProvenance|create_provenance" kosmos/orchestration/delegation.py` returns zero matches. The DelegationManager does not import or reference any provenance or notebook generation modules.

This means the provenance pipeline infrastructure exists (provenance.py, notebook_generator.py, Finding.code_provenance field) but is not wired into the live orchestration path through DelegationManager. Provenance attachment would need to happen either in the delegation layer after task execution or within individual agent implementations.

**Terminal Side Effects:**

| Side Effect | Location | Mechanism |
|---|---|---|
| Docker container lifecycle | sandbox.py:285-400 | create -> start -> wait -> logs -> remove |
| Code exec() in restricted namespace | executor.py:617 | `exec(code, exec_globals, exec_locals)` |
| Database INSERT for protocol | experiment_designer.py:891-906 | SQLAlchemy `session.add()` + `commit()` |
| Notebook .ipynb written to disk | notebook_generator.py:273-274 | `nbformat.write(nb, f)` |
| Finding JSON artifact saved | artifacts.py:488-492 | `summary_path.write()` |
| Container resource stats streamed | sandbox.py:402-436 | `container.stats(stream=True)` |

**Security Layers (Defense in Depth):**

1. **CodeValidator** gate: AST-based import/pattern/ethical checks. Blocks execution entirely on violation [FACT] (executor.py:1040-1048).
2. **Restricted builtins**: Whitelist of safe builtins with restricted `__import__` for direct execution [FACT] (executor.py:43-110).
3. **Docker sandbox**: Network-disabled, read-only filesystem, capability-dropped, resource-limited container [FACT] (sandbox.py:265-275).
4. **Timeout protection**: 300-second default on both sandbox and direct paths [FACT] (executor.py:41, sandbox.py:84).

---

### Path 3: P1.4: Literature Pipeline Trace

```
UnifiedLiteratureSearch.search() (kosmos/literature/unified_search.py:23)
  [FACT] Fans out queries to three clients in parallel via ThreadPoolExecutor
  (unified_search.py:147) with a configurable timeout (unified_search.py:164).
  Results are deduplicated across sources using DOI > arXiv > PubMed > title priority
  (unified_search.py:383-428), then ranked by scoring formula combining citation count
  (max 100 pts), title relevance (max 50 pts), abstract relevance (max 30 pts), and
  recency (max 20 pts) (unified_search.py:475-497).
  When extract_full_text=True, calls PDFExtractor.extract_paper_text() for each paper
  with a per-paper timeout enforced via a single-worker ThreadPoolExecutor
  (unified_search.py:517-523).

  -> ArxivHTTPClient.search() (kosmos/literature/arxiv_http_client.py:137) (see Module Index: base_client.py)
  |    [SIDE EFFECT: HTTP request to http://export.arxiv.org/api/query]
  |    [FACT] Uses httpx.Client (arxiv_http_client.py:105). Rate limiting enforces a 3-second
  |    minimum interval between requests (arxiv_http_client.py:34,
  |    ARXIV_RATE_LIMIT_SECONDS = 3.0).
  |    -> Response parsed: Atom 1.0 XML parsed by xml.etree.ElementTree
  |    |    [FACT] (arxiv_http_client.py:416, ET.fromstring(xml_content)).
  |    -> Data transformation: Raw XML -> ArxivSearchResult dataclass (intermediate, line 44-58)
  |    |    -> PaperMetadata via _result_to_metadata() (arxiv_http_client.py:543-576).
  |    |    Conversion normalizes authors from strings to Author objects, maps categories to
  |    |    fields, and extracts pdf_url from Atom <link> elements with title="pdf"
  |    |    [FACT] (arxiv_http_client.py:494-497).
  |    |
  |    | [FACT] Key limitation: get_paper_references() and get_paper_citations() both return
  |    | empty lists (arxiv_http_client.py:325-326, line 343-344). Citation data requires
  |    | Semantic Scholar.
  |    |
  |    | [FACT] Client caches results via kosmos.literature.cache.get_cache()
  |    | (arxiv_http_client.py:114).
  |
  -> PDFExtractor.extract_paper_text() (kosmos/literature/pdf_extractor.py:186)
  |    [SIDE EFFECT: HTTP request to download PDF]
  |    Takes a PaperMetadata object, uses its pdf_url to download the PDF via httpx.Client
  |    [FACT] (pdf_extractor.py:231), then extracts text using PyMuPDF (fitz)
  |    [FACT] (pdf_extractor.py:8, import fitz).
  |    Downloaded PDFs are cached on disk at .pdf_cache/{paper_id}.pdf
  |    [FACT] (pdf_extractor.py:88-90).
  |    -> Data transformation: Raw PDF bytes -> per-page text via page.get_text()
  |    |    [FACT] (pdf_extractor.py:287) -> joined with double newlines
  |    |    (pdf_extractor.py:290) -> cleaned by _clean_text() which collapses whitespace,
  |    |    removes page numbers, strips non-ASCII, and normalizes line breaks
  |    |    [FACT] (pdf_extractor.py:334-356).
  |    |
  |    | [FACT] Fallback behavior: If PDF extraction fails or no URL is available, falls back
  |    | to paper.abstract (pdf_extractor.py:216-218). A minimum threshold of 100 characters
  |    | determines if extracted text is too short (indicating a scanned/image-based PDF)
  |    | (pdf_extractor.py:296-298).
  |    |
  |    | [FACT] Result stored directly on paper.full_text (mutation in place)
  |    | (pdf_extractor.py:213).
  |    |
  |    | Singleton access: get_pdf_extractor() at line 388 provides module-level singleton.
  |
  -> ReferenceManager.add_reference() (kosmos/literature/reference_manager.py:69)
  |    Stores references in an in-memory dict (self.references: Dict[str, PaperMetadata])
  |    keyed by generated IDs [FACT] (reference_manager.py:55). IDs generated from DOI,
  |    arXiv ID, PubMed ID, or title hash (md5, first 8 chars) [FACT]
  |    (reference_manager.py:367-380).
  |    -> Deduplication: When auto_deduplicate=True (default), each add_reference() call
  |    |    checks existing refs via DeduplicationEngine.is_duplicate()
  |    |    [FACT] (reference_manager.py:85-89). Engine uses priority chain: DOI exact match
  |    |    -> arXiv ID exact match -> PubMed ID exact match -> fuzzy title match at
  |    |    threshold 0.9 using SequenceMatcher [FACT] (reference_manager.py:732-764).
  |    |    On duplicate, metadata is merged (preferring non-empty fields, higher citation
  |    |    counts, more recent years) [FACT] (reference_manager.py:674-730).
  |    |
  |    | [FACT] Persistence: Optionally serialized to JSON on disk via _save_to_storage()
  |    | (reference_manager.py:396-424). Citation links (citing_id -> [cited_ids]) tracked
  |    | separately in self.citation_links (reference_manager.py:58).
  |    |
  |    | Export: Delegates to citations.py for BibTeX and RIS export via CitationFormatter
  |    | [FACT] (reference_manager.py:322-323, importing papers_to_bibtex, papers_to_ris).
  |
  -> GraphBuilder.add_paper() (kosmos/knowledge/graph_builder.py:86) — terminal side effect (see Module Index: knowledge_graph.py)
       Coordinates three sub-operations per paper:
       |
       -> KnowledgeGraph.create_paper() [FACT] (graph.py:204) — Paper node creation
       |    Creates Paper node in Neo4j via py2neo [FACT] (graph.py:15,
       |    from py2neo import Graph, Node, Relationship). Properties include id, title,
       |    abstract, year, citation_count, domain (first field), plus optional
       |    DOI/arXiv/PubMed identifiers [FACT] (graph.py:226-244). Merge-on-id is default
       |    behavior (upsert) [FACT] (graph.py:246-254).
       |    [SIDE EFFECT: Neo4j node creation]
       |
       -> _add_paper_authors() [FACT] (graph_builder.py:205) — Author relationships
       |    Creates Author nodes and AUTHORED relationships with order and role
       |    (first author / corresponding) properties [FACT] (graph_builder.py:227-234).
       |    [SIDE EFFECT: Neo4j node + relationship creation]
       |
       -> _extract_and_add_concepts() [FACT] (graph_builder.py:241) — Claude-powered
       |    -> ConceptExtractor.extract_from_paper() [FACT] (concept_extractor.py:152)
       |    |    -> self.client.messages.create() [FACT] (concept_extractor.py:294-304)
       |    |    [SIDE EFFECT: Claude API call]
       |    |    Claude returns structured JSON with concepts (name, description, domain,
       |    |    relevance) and methods (name, description, category, confidence)
       |    |    [FACT] (concept_extractor.py:421-451).
       |    -> Concept and Method nodes connected by DISCUSSES and USES_METHOD relationships
       |    |    [FACT] (graph_builder.py:259-309).
       |    -> Concept-to-concept RELATED_TO relationships also extracted and created
       |         [FACT] (graph_builder.py:312-325).
       |    [SIDE EFFECT: Neo4j nodes + relationships creation]
       |
       -> _add_paper_citations() [FACT] (graph_builder.py:330) — Citation edges
            Creates CITES relationships between paper nodes, but only if the cited paper
            already exists in the graph [FACT] (graph_builder.py:343-353).
            [SIDE EFFECT: Neo4j relationship creation]
```

**Citation Network: Graph-Side Analysis:**

`CitationNetwork` in `kosmos/literature/citations.py:608` builds NetworkX-based citation graphs. It optionally integrates with the Neo4j knowledge graph [FACT] (citations.py:625-628, `from kosmos.knowledge.graph import get_knowledge_graph`) to pull `CITES` edges when `use_knowledge_graph=True`. The class provides PageRank, betweenness centrality, shortest-path, and seminal-paper identification [FACT] (citations.py:711-797).

**Neo4j Schema -- Node Types:**

| Node Type | Indexes | Key Properties |
|-----------|---------|----------------|
| Paper | id, doi, arxiv_id, pubmed_id | title, abstract, year, citation_count, domain |
| Author | name | affiliation, h_index, paper_count |
| Concept | name, domain | description, frequency |
| Method | name, category | description, usage_count |

[FACT: graph.py:176-191 defines all 8 indexes]

| Relationship | Source -> Target | Key Properties |
|-------------|-----------------|----------------|
| CITES | Paper -> Paper | created_at |
| AUTHORED | Author -> Paper | order, role |
| DISCUSSES | Paper -> Concept | relevance_score, section |
| USES_METHOD | Paper -> Method | confidence, context |
| RELATED_TO | Concept -> Concept | similarity, source |

[FACT: graph.py:541-745 defines all 5 relationship types]

**Full Data Transformation Summary:**

```
arXiv XML Atom feed
  |  [ArxivHTTPClient._parse_atom_feed]
  v
ArxivSearchResult (dataclass, 12 fields)
  |  [ArxivHTTPClient._result_to_metadata]
  v
PaperMetadata (dataclass, 20+ fields, universal schema)
  |  [PDFExtractor.extract_paper_text]  -- mutates paper.full_text in place
  v
PaperMetadata + full_text (raw text from PDF, cleaned)
  |  [ReferenceManager.add_reference]  -- dedup, merge, persist
  v
Dict[ref_id, PaperMetadata] (deduplicated collection)
  |  [GraphBuilder.add_paper]
  |    +-- KnowledgeGraph.create_paper       -> Paper node in Neo4j
  |    +-- KnowledgeGraph.create_author      -> Author node + AUTHORED rel
  |    +-- ConceptExtractor.extract_from_paper -> Claude API call
  |    +-- KnowledgeGraph.create_concept     -> Concept node + DISCUSSES rel
  |    +-- KnowledgeGraph.create_method      -> Method node + USES_METHOD rel
  |    +-- KnowledgeGraph.create_related_to  -> RELATED_TO between concepts
  v
Neo4j graph database (terminal side effect)
```

**Singleton Pattern:**

[PATTERN: 7/7 modules checked] Every major component uses the singleton-with-reset pattern:
- `get_pdf_extractor()` / `reset_extractor()` [FACT: pdf_extractor.py:388-407]
- `get_reference_manager()` / `reset_reference_manager()` [FACT: reference_manager.py:783-811]
- `get_knowledge_graph()` / `reset_knowledge_graph()` [FACT: graph.py:1003-1037]
- `get_graph_builder()` / `reset_graph_builder()` [FACT: graph_builder.py:505-533]
- `get_concept_extractor()` / `reset_concept_extractor()` [FACT: concept_extractor.py:623-651]

**Bug: PaperSource.OTHER Does Not Exist:** (see Gotcha #73)

`citations.py` references `PaperSource.OTHER` at lines 206 and 262 for BibTeX/RIS parsed entries. However, the `PaperSource` enum in `base_client.py:17-23` only defines `ARXIV`, `SEMANTIC_SCHOLAR`, `PUBMED`, `UNKNOWN`, and `MANUAL`. There is no `OTHER` variant. This will raise an `AttributeError` at runtime when parsing BibTeX or RIS files. [FACT: citations.py:206, citations.py:262 reference `PaperSource.OTHER`; base_client.py:17-23 omits it]. [ABSENCE: confirmed via grep -- `PaperSource.OTHER` appears only in citations.py, never defined].

---

### Path 4: WebSocket Event Streaming Pipeline

```
[Three independent event producers]

Source A: AnthropicProvider.generate_stream_async() (kosmos/core/providers/anthropic.py:772)
  [FACT] The synchronous generator generate_stream() at line 680 uses event_bus.publish_sync()
  (anthropic.py:710,736,749,763), while the async generator generate_stream_async() at line 772
  uses await event_bus.publish() (anthropic.py:817,843,856,870).
  Both emit a three-phase lifecycle:
    LLM_CALL_STARTED before streaming begins
    LLM_TOKEN for each text chunk yielded from stream.text_stream
    LLM_CALL_COMPLETED after the stream closes (with duration_ms and truncated content[:200])
  x on exception: LLM_CALL_FAILED is emitted instead.
  [ABSENCE] The LiteLLM provider does NOT emit any streaming events -- this is an asymmetry
  only the Anthropic provider covers.

Source B: ResearchLoop (kosmos/workflow/research_loop.py)
  ResearchLoop.__init__() at line 142-153 generates a process_id as "research_{uuid4_hex[:8]}"
  and obtains the singleton EventBus via get_event_bus()
  [FACT] (research_loop.py:143-150).
  run() emits:
    WorkflowEvent(WORKFLOW_STARTED) before the cycle loop [FACT] (research_loop.py:187)
    CycleEvent(CYCLE_STARTED/COMPLETED/FAILED) at cycle boundaries
      [FACT] (research_loop.py:197,215,239)
    WorkflowEvent(WORKFLOW_PROGRESS) after each successful cycle with cumulative
      progress_percent and findings_count [FACT] (research_loop.py:227)
    WorkflowEvent(WORKFLOW_COMPLETED) after final statistics computation
      [FACT] (research_loop.py:251)
  All use await self._event_bus.publish(event) -- the async path. Guard checks
  self._emit_events and self._event_bus is not None at the top of each emitter to degrade
  gracefully if the event bus is unavailable [FACT] (research_loop.py:508,560).

Source C: StageTracker (kosmos/core/stage_tracker.py)
  _publish_to_event_bus() method at line 186 converts the tracker's own StageEvent into
  the streaming module's StageEvent (aliased as StreamingStageEvent) and calls
  get_event_bus().publish_sync() [FACT] (stage_tracker.py:215). This is synchronous because
  StageTracker.track() is a @contextmanager, not async. The tracker's emit_to_event_bus flag
  (default True) at line 61 controls whether events are forwarded
  [FACT] (stage_tracker.py:61,71).

[All sources converge on the EventBus]

  -> EventBus.publish() / publish_sync() (kosmos/core/event_bus.py:125,154) (see Gotcha #130)
       [FACT] Singleton at kosmos/core/event_bus.py obtained via get_event_bus() at line 264,
       which lazily creates the global _event_bus instance (event_bus.py:264-274).
       Maintains two data structures:
         _subscribers: Dict[Optional[EventType], List[Callback]] for type-routed subscriptions
           (where None key means "all events") [FACT] (event_bus.py:51)
         _process_filters: Dict[Callback, Optional[Set[str]]] for process ID filtering
           [FACT] (event_bus.py:53)

       -> _get_callbacks() (line 192) — collects both global subscribers (_subscribers[None])
       |    and type-specific subscribers (_subscribers[event.type]).
       -> _passes_process_filter() (line 205) — checks whether the event's process_id is in
            the callback's filter set (or passes if the filter is None, meaning accept all)
            [FACT] (event_bus.py:192-221).

       Dual publish paths:
         publish() (async, line 125): awaits async callbacks and calls sync callbacks directly.
         publish_sync() (sync, line 154): calls sync callbacks directly but schedules async
           callbacks via loop.create_task() if a running event loop exists -- otherwise
           silently skips them [FACT] (event_bus.py:178-184). This means the stage tracker's
           synchronous publishes can still reach async WebSocket subscribers, but only when
           an event loop is already running.

       Thread safety: Both paths acquire self._sync_lock (a threading.Lock) around
         _get_callbacks() [FACT] (event_bus.py:55-56,138,167). Error handling wraps each
         callback invocation individually with logger.error() -- one failing subscriber does
         not block others [FACT] (event_bus.py:151-152).

  -> queue_callback() [per-connection closure] (kosmos/api/websocket.py:189)
  |    await queue.put(event)
  |
  -> asyncio.Queue per WebSocket connection (kosmos/api/websocket.py:187)
  |
  -> WebSocket endpoint mounted at /ws/events via APIRouter(prefix="/ws")
       [FACT] (websocket.py:31,123). websocket_events() handler at line 124 accepts optional
       process_id and types query parameters for initial filtering.

       Connection lifecycle:
       1. manager.connect(websocket) accepts the WebSocket and adds it to
          ConnectionManager.active_connections: Set[WebSocket] [FACT] (websocket.py:44-45,174).
       2. Per-connection asyncio.Queue[StreamingEvent] created at line 187
          [FACT] (websocket.py:187).
       3. Async queue_callback closure at line 189 subscribed to the event bus with parsed
          filters [FACT] (websocket.py:189-195).
       4. Subscription confirmation JSON sent immediately [FACT] (websocket.py:198-202).
       5. Two concurrent tasks created: send_events() and receive_messages()
          [FACT] (websocket.py:281-282).

       -> send_events() task (line 204):
       |    Loops indefinitely, pulling events from the queue with await queue.get(),
       |    serializing via json.dumps(asdict(event), default=str), and sending as text
       |    with websocket.send_text(data) [FACT] (websocket.py:208-210).
       |    Breaks on CancelledError or any send exception.
       |    [SIDE EFFECT: WebSocket frame sent to client]

       -> receive_messages() task (line 217):
            Processes client commands:
            "ping" -> responds with {"action": "pong", "timestamp": ...}
              [FACT] (websocket.py:227-233).
            "subscribe" -> unsubscribes old callback, re-subscribes with new filters,
              sends confirmation [FACT] (websocket.py:235-265). This enables dynamic filter
              changes without reconnecting.
            Invalid JSON -> sends {"action": "error", "message": "Invalid JSON"}
              [FACT] (websocket.py:272-274).

       Teardown: asyncio.wait(FIRST_COMPLETED) at line 286 terminates when either task exits
       (typically receive_messages on disconnect). Pending tasks are cancelled. The finally
       block disconnects the websocket from the manager and unsubscribes from the event bus
       [FACT] (websocket.py:302-303).
```

**SSE Parallel Path (Comparison):**

The SSE endpoint at `kosmos/api/streaming.py` provides a one-directional alternative. It mounts at `/stream/events` [FACT] (streaming.py:31,104), uses the same `asyncio.Queue` + `queue_callback` pattern [FACT] (streaming.py:52-56), but formats output as Server-Sent Events (`event: {type}\ndata: {json}\n\n`) [FACT] (streaming.py:90) and includes a keepalive heartbeat via `asyncio.wait_for` timeout (default 30 seconds) [FACT] (streaming.py:82-94). The SSE path uses FastAPI's `StreamingResponse` with `text/event-stream` media type and disables nginx buffering via `X-Accel-Buffering: no` [FACT] (streaming.py:158-166). Unlike WebSocket, clients cannot dynamically change their subscription filters.

**CLI Consumer Path:**

`StreamingDisplay` in `kosmos/cli/streaming.py` provides a Rich-based terminal consumer. It subscribes with 12 specific `EventType` values [FACT] (cli/streaming.py:116-129) and routes events through a handler dispatch table at line 163 [FACT] (cli/streaming.py:163-176). Token events are buffered in `_current_tokens` with a configurable `max_token_display` cap (default 500) [FACT] (cli/streaming.py:255-259). A `SimpleStreamingDisplay` fallback exists for environments without Rich [FACT] (cli/streaming.py:309).

**Event Type Taxonomy:**

The `EventType` enum at `kosmos/core/events.py:16` defines 21 event types across 6 categories [FACT] (events.py:16-52):

| Category | Types | Dataclass |
|----------|-------|-----------|
| Workflow | started, progress, completed, failed | `WorkflowEvent` |
| Cycle | started, completed, failed | `CycleEvent` |
| Task | generated, started, completed, failed | `TaskEvent` |
| LLM | call_started, token, call_completed, call_failed | `LLMEvent` |
| Execution | validating, executing, output, completed, failed | `ExecutionEvent` |
| Stage | started, completed, failed | `StageEvent` |

`StreamingEvent` is a `Union` of all 7 event dataclasses [FACT] (events.py:216-224). All share `BaseEvent` fields: `type`, `timestamp` (auto-generated UTC ISO), `process_id`, and `correlation_id` [FACT] (events.py:60-69).

**Error Handling Summary:**

- **Producer side**: All event emission is wrapped in try/except with `logger.debug()` -- emission failures never crash the workflow [FACT] (research_loop.py:535-536, stage_tracker.py:219-220).
- **EventBus**: Each callback is invoked in its own try/except; one subscriber failure does not affect others [FACT] (event_bus.py:151-152).
- **WebSocket send**: `ConnectionManager.send_event()` catches all exceptions and removes dead connections [FACT] (websocket.py:83-90). `broadcast()` collects disconnected clients into a list and removes them after iteration [FACT] (websocket.py:100-110).
- **WebSocket receive**: `WebSocketDisconnect` and `json.JSONDecodeError` have dedicated handlers; generic exceptions break the loop [FACT] (websocket.py:269-278).

**Async Patterns and Concurrency Considerations:**

The `websocket_events()` handler uses the `asyncio.wait(FIRST_COMPLETED)` pattern for bidirectional communication -- two coroutines run concurrently and the first one to exit triggers cleanup of the other [FACT] (websocket.py:286-298). Each WebSocket connection gets its own `asyncio.Queue`, providing back-pressure isolation between fast publishers and slow clients -- events queue up rather than being dropped. However, there is no queue size limit configured, meaning a slow client could accumulate unbounded memory in its queue (see Gotcha #129). The `ConnectionManager` uses a `Set[WebSocket]` for O(1) add/remove/membership and iterates a copy (`list(self.active_connections)`) during broadcast to avoid mutation during iteration [FACT] (websocket.py:102).
## Module Behavioral Index

### `alerts.py` -- Detailed Behavioral Analysis

**Module**: `kosmos/monitoring/alerts.py` (570 lines)

**What it does**: Alerting system for critical events in the Kosmos AI Scientist platform. Provides rule-based alert evaluation, lifecycle management (active/acknowledged/resolved), and notification dispatch to multiple channels (logging, email/SMTP, Slack webhook, PagerDuty Events API v2). Re-exported via `kosmos/monitoring/__init__.py:14-24` -- all public symbols are available at the package level.

**Git Risk**: risk=0.24, churn=2, hotfixes=1, authors=2 -- Low risk, infrequent changes.

**What's non-obvious**:

1. `Alert.__post_init__()` auto-generates `alert_id` as `"{name}_{unix_timestamp}"` if not provided. Two alerts of the same name triggered within the same second get the same ID, which can cause collisions in `AlertManager.active_alerts` dict. The second alert would silently overwrite the first.

2. `AlertRule.should_trigger()` evaluates the condition callable inside a try/except. If the condition callable raises, it catches the exception and returns False -- meaning a broken checker is silently treated as "no alert needed".

3. Three of seven default alert rules are placeholders that never fire: `high_api_failure_rate`, `api_rate_limit_warning`, and `high_experiment_failure_rate` always return False.

4. Two health-checker-based rules (`database_connection_failed`, `cache_unavailable`) perform lazy imports of `kosmos.api.health` inside the check method, meaning the health module is only loaded when rule evaluation actually runs. This creates a bidirectional dependency between monitoring and API layers.

5. The module-level singleton `get_alert_manager()` creates the singleton on first call. Registers `log_notification_handler` unconditionally, then conditionally registers email/Slack/PagerDuty handlers based on environment variables at creation time. Handler registration happens once at singleton creation. If environment variables change after first call, the handler set does not update.

**What breaks if you change it**:

- External I/O: Can send emails, Slack messages, and PagerDuty incidents -- three distinct external system integrations, all gated by environment variables.
- Health check coupling: Two rules import `kosmos.api.health` at runtime, creating a bidirectional dependency between monitoring and API layers.
- Memory: `alert_history` grows up to 1000 entries in memory with no persistence.
- No thread safety: `AlertManager` uses no locks. Concurrent calls to `evaluate_rules()` and `resolve_alert()` could produce inconsistent state in `active_alerts`.

**Runtime Dependencies**:

- Standard library: `logging`, `os`, `json`, `time`, `datetime`, `enum`, `dataclasses`
- External (conditional): `psutil` for memory/disk checks (lines 299, 310), `requests` for Slack (line 419) and PagerDuty (line 490) notifications, `smtplib` for email (line 366). These are imported inside function bodies, so import failure is deferred to call time.

**Public API**:

- **`AlertSeverity(str, Enum)`**: Four levels: `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Inherits from `str` for JSON-friendly serialization.
- **`AlertStatus(str, Enum)`**: Three states: `ACTIVE`, `RESOLVED`, `ACKNOWLEDGED`.
- **`Alert` (dataclass)**: Fields: `name` (str), `severity` (AlertSeverity), `message` (str), `timestamp` (datetime, default `datetime.utcnow()`), `status` (AlertStatus, default ACTIVE), `details` (Dict, default empty), `alert_id` (Optional[str]). `to_dict()` serializes all fields, converting enum values via `.value` and timestamp via `.isoformat()`.
- **`AlertRule` (dataclass)**: Defines trigger conditions. Fields: `name` (str), `condition` (Callable[[], bool]), `severity` (AlertSeverity), `message_template` (str), `cooldown_seconds` (int, default 300 = 5 minutes), `last_triggered` (Optional[datetime]). `should_trigger()` checks cooldown period against `datetime.utcnow()`, then evaluates the condition callable. If the condition callable raises, it catches the exception and returns False. `trigger(details)` updates `last_triggered` timestamp and returns a new `Alert` object.
- **`AlertManager.add_alert_rule(rule)`**: Appends a rule. No dedup check -- the same rule name can be added multiple times.
- **`AlertManager.add_notification_handler(handler)`**: Appends a handler callable. Logs the handler's `__name__`, so lambdas will log as `<lambda>`.
- **`AlertManager.evaluate_rules()`**: Iterates all rules, calls `should_trigger()`, and if True, calls `trigger()` and `_handle_alert()`. Synchronous, serial evaluation -- no parallelism.
- **`AlertManager.resolve_alert(alert_id)`**: Marks alert as RESOLVED, removes from `active_alerts`. Does nothing if `alert_id` not found (no error raised).
- **`AlertManager.acknowledge_alert(alert_id)`**: Marks alert as ACKNOWLEDGED but does NOT remove from `active_alerts`.
- **`AlertManager.get_active_alerts()`**: Returns a list copy of active alert values.
- **`AlertManager.get_alert_history(limit=100)`**: Returns last `limit` alerts from history.
- **`AlertManager._handle_alert(alert)`**: Stores alert in `active_alerts` and appends to `alert_history`. Trims history to 1000 entries when it exceeds that count (hard-coded limit). Then dispatches to all notification handlers in a try/except -- a failing handler does not prevent subsequent handlers from running.
- **`log_notification_handler(alert)`**: Always-on handler. Maps severity to the corresponding `logger` method (info/warning/error/critical). Formats as `ALERT [SEVERITY]: message | details`.
- **`email_notification_handler(alert)`**: Gated by `ALERT_EMAIL_ENABLED` env var (default "false"). Uses SMTP with STARTTLS. Reads 6 environment variables: `ALERT_EMAIL_ENABLED`, `ALERT_EMAIL_TO`, `ALERT_EMAIL_FROM` (defaults to `alerts@kosmos.ai`), `SMTP_HOST` (defaults to `localhost`), `SMTP_PORT` (defaults to 587), `SMTP_USER`, `SMTP_PASSWORD`. Uses `smtplib.SMTP` with `starttls()` -- this assumes the server supports STARTTLS on the given port. If `SMTP_PORT` is 465 (implicit TLS), this will likely fail because `SMTP` (not `SMTP_SSL`) is used. On failure, logs error and returns silently.
- **`slack_notification_handler(alert)`**: Gated by `ALERT_SLACK_ENABLED` env var. Sends a Slack webhook payload with color-coded attachments (green/orange/red/dark-red by severity). Uses `requests.post()` with 10-second timeout. On failure, logs error and returns.
- **`pagerduty_notification_handler(alert)`**: Gated by `ALERT_PAGERDUTY_ENABLED` env var. Only triggers for ERROR and CRITICAL severities -- INFO and WARNING are silently dropped. Posts to PagerDuty Events API v2 at `https://events.pagerduty.com/v2/enqueue`. Uses `alert.alert_id` as `dedup_key` for PagerDuty deduplication. On failure, logs error and returns.
- **`get_alert_manager()`**: Singleton accessor. Creates manager on first call, registers handlers based on env vars. Subsequent calls return same instance regardless of env var changes.
- **`evaluate_alerts()`**: Convenience function wrapping `get_alert_manager().evaluate_rules()`.
- **`get_active_alerts()`**: Returns serialized (dict) list of active alerts.
- **`get_alert_history(limit=100)`**: Returns serialized (dict) list of history.

**Test Coverage**: No dedicated test file exists for `alerts.py`. No file matching `test*alert*` was found under `tests/`. The module is exported from `kosmos/monitoring/__init__.py` but has zero automated tests.

**Gotchas**:

1. [MEDIUM] **Alert ID collisions**: Same-name alerts triggered within one second share an ID, causing silent overwrites in `active_alerts`.
2. [MEDIUM] **SMTP hardcoded to STARTTLS**: Port 465 (implicit TLS) connections will fail; only port 587 pattern works.
3. [MEDIUM] **PagerDuty silently drops non-ERROR/CRITICAL**: This filtering is not documented in the function's env var docs.
4. [LOW] **3 of 7 default rules are placeholders**: `high_api_failure_rate`, `api_rate_limit_warning`, and `high_experiment_failure_rate` always return False.
5. [MEDIUM] **Singleton handler registration is frozen at creation time**: Changing env vars after first `get_alert_manager()` call has no effect.
6. [MEDIUM] **No thread safety on AlertManager**: No locking on `active_alerts` or `alert_history` mutation.
7. [HIGH] **No test coverage**: Zero automated tests for this module.

---

### `base_client.py` -- Detailed Behavioral Analysis

**Module**: `kosmos/literature/base_client.py`

**What it does**: Defines the abstract base class (`BaseLiteratureClient`) and unified data models (`PaperMetadata`, `Author`, `PaperSource`) for all literature API clients in the system. Every literature source -- arXiv, Semantic Scholar, PubMed -- inherits from `BaseLiteratureClient` and must implement the four abstract methods: `search()`, `get_paper_by_id()`, `get_paper_references()`, and `get_paper_citations()`. This module establishes the contract that normalizes heterogeneous academic API responses into a single `PaperMetadata` dataclass.

**Git Risk**: risk=0.24, churn=2, hotfixes=1, authors=2 -- Low risk, stable contract.

**What's non-obvious**:

1. `PaperMetadata` is a dataclass, not Pydantic. Unlike the hypothesis models which use Pydantic `BaseModel`, `PaperMetadata` uses `@dataclass`. This means no automatic validation, no JSON schema generation, and no `model_dump()`/`model_validate()` methods. The `to_dict()` method is hand-written and does not handle all edge cases (e.g., `raw_data` is excluded from serialization).

2. Mutable default arguments handled via `__post_init__`. The dataclass fields `authors`, `fields`, and `keywords` default to `None` rather than `[]`, with `__post_init__` replacing `None` with empty lists. This avoids the classic mutable default argument bug, but means type hints (`List[Author]`) are technically incorrect since the field can be `None` before `__post_init__` runs.

3. `_validate_query()` warns on long queries but does NOT truncate. The method logs a warning for queries over 1000 characters and returns `True`, meaning validation passes even for overlong queries. The comment says "truncating to 1000" but the code does not actually truncate. This is a misleading log message.

4. `_normalize_paper_metadata()` raises NotImplementedError rather than being decorated `@abstractmethod`. This means a subclass that forgets to override it will fail at runtime rather than at class definition time.

5. `PaperSource` enum includes `MANUAL` value beyond the three API sources (arXiv, Semantic Scholar, PubMed), plus `UNKNOWN`. The `MANUAL` source enables papers added by hand, not fetched from any API.

6. `primary_identifier` property has a priority chain: Returns DOI first, then arXiv ID, then PubMed ID, then source-specific ID. This means a paper with both a DOI and arXiv ID will always identify as its DOI.

7. `raw_data` field stores the full API response for debugging but can be very large (entire JSON responses). It is not included in `to_dict()` output, meaning it is lost on serialization.

**What breaks if you change it**:

Imported by 18 files across the codebase (20 occurrences): All literature client implementations (`arxiv_client.py`, `arxiv_http_client.py`, `semantic_scholar.py`, `pubmed_client.py`, `pdf_extractor.py`, `citations.py`, `reference_manager.py`, `unified_search.py`, `cache.py`), knowledge modules (`vector_db.py`, `semantic_search.py`, `graph_builder.py`, `graph.py`, `concept_extractor.py`, `embeddings.py`), agent modules (`literature_analyzer.py`, `hypothesis_generator.py`), world model (`world_model/simple.py`), hypothesis (`novelty_checker.py`). Changing `PaperMetadata` fields or `BaseLiteratureClient` method signatures would require updates across all these modules.

**Runtime Dependencies**:

- `abc` (ABC, abstractmethod)
- `typing`, `dataclasses`, `datetime`, `enum` (stdlib)
- `logging` (stdlib)
- Zero external dependencies. This is a pure contract/interface module.

**Public API**:

- **`class PaperSource(str, Enum)`**: Values: `ARXIV`, `SEMANTIC_SCHOLAR`, `PUBMED`, `UNKNOWN`, `MANUAL`.
- **`class Author`**: Fields: `name` (required), `affiliation` (optional), `email` (optional), `author_id` (optional). Plain dataclass with no validation.
- **`class PaperMetadata`**: Identifier fields: `id` (required, source-specific), `source` (PaperSource), `doi`, `arxiv_id`, `pubmed_id`. Core fields: `title`, `abstract`, `authors`. Publication fields: `publication_date`, `journal`, `venue`, `year`. Link fields: `url`, `pdf_url`. Citation fields: `citation_count`, `reference_count`, `influential_citation_count`. Content fields: `fields`, `keywords`, `full_text`. Debug field: `raw_data`. `primary_identifier` property: DOI > arXiv > PubMed > source ID. `author_names` property: Extracts name strings from Author objects. `to_dict()`: Serializes to dict for DB storage. Converts `publication_date` to ISO format, authors to dicts.
- **`class BaseLiteratureClient(ABC)`**: `__init__(api_key, cache_enabled)` stores API key, cache flag, and creates a logger namespaced to the subclass name. `search(query, max_results=10, fields, year_from, year_to, **kwargs) -> List[PaperMetadata]`: Abstract. Must return list of `PaperMetadata` matching the query. `get_paper_by_id(paper_id) -> Optional[PaperMetadata]`: Abstract. Returns single paper or None. `get_paper_references(paper_id, max_refs) -> List[PaperMetadata]`: Abstract. Returns papers cited by the given paper. `get_paper_citations(paper_id, max_cites) -> List[PaperMetadata]`: Abstract. Returns papers citing the given paper. `get_source_name() -> str`: Returns class name with "Client" suffix removed. `_handle_api_error(error, operation)`: Logs error with `exc_info=True` for full traceback. Placeholder for future retry logic. Does not re-raise or return anything. `_validate_query(query) -> bool`: Returns `False` for empty/whitespace queries. Returns `True` for all others including overlong queries. `_normalize_paper_metadata(raw_data) -> PaperMetadata`: Raises `NotImplementedError`. Not an `@abstractmethod` -- subclasses can be instantiated without overriding it.

**Test Coverage**: Literature client tests found: `tests/unit/literature/test_arxiv_client.py`, `tests/unit/literature/test_arxiv_http_client.py`, `tests/unit/literature/test_semantic_scholar.py`, `tests/unit/literature/test_pubmed_client.py`, `tests/unit/literature/test_unified_search.py`, `tests/unit/literature/test_citations.py`. Requirements tests: `tests/requirements/literature/test_req_literature.py`. **Gap**: No direct unit test for `base_client.py` itself. The abstract methods are tested indirectly through concrete implementations. No test validates `PaperMetadata.to_dict()`, `_validate_query()` behavior for overlong queries, or `primary_identifier` priority chain.

**Gotchas**:

1. [MEDIUM] **`_validate_query()` logs "truncating" but does not truncate**: Callers that trust the validation may send overlong queries to APIs that have their own length limits.
2. [MEDIUM] **`_normalize_paper_metadata()` is not `@abstractmethod`**: A subclass can forget to implement it and only discover the error at runtime when a search result is being processed.
3. [LOW] **`raw_data` is excluded from `to_dict()`**: Any code that round-trips through `to_dict()` loses the raw API response. This is intentional (for DB storage efficiency) but could surprise debugging workflows.
4. [LOW] **`PaperMetadata.authors` type hint is `List[Author]` but defaults to `None`**: Between construction and `__post_init__`, the field is `None`, not a list. Code that accesses `paper.authors` in a custom `__init__` before `__post_init__` runs will get `None`.
5. [LOW] **No rate limiting built into the base class**: The `_handle_api_error()` method is the only error handling hook, and it only logs. Subclasses must implement their own rate limiting, retry, and circuit breaker logic.
6. [LOW] **`to_dict()` does not handle `None` authors gracefully**: If `authors` is somehow still `None`, the list comprehension would raise `TypeError`.

---

### `cli_main.py` -- Detailed Behavioral Analysis

**Module**: `kosmos/cli/main.py` (441 lines)

**What it does**: CLI entry point for the Kosmos AI Scientist. Builds a Typer application with global options, inline commands (`version`, `info`, `doctor`), and dynamically registered sub-commands from `kosmos.cli.commands`. Package entry point registered in `pyproject.toml:179` as `kosmos = "kosmos.cli.main:cli_entrypoint"`.

**Git Risk**: risk=0.63, churn=10, hotfixes=6, authors=2 -- Moderate-high risk, frequent changes with many hotfixes.

**What's non-obvious**:

1. **Four side effects at import time**: (a) `load_dotenv()` loads `.env` file into environment immediately when the module is imported -- importing `kosmos.cli.main` in a test mutates `os.environ`. (b) `install_rich_traceback()` installs a global Rich traceback handler with `show_locals=False, width=120`, overriding Python's default exception display for the entire process. (c) `app = typer.Typer(...)` creates the global Typer app instance with `no_args_is_help=True` and `add_completion=False`. (d) `register_commands()` called at module level imports 7 command modules and registers them. On `ImportError`, silently passes -- commands simply do not appear.

2. **Database initialization on every invocation**: The `main()` callback initializes the database via `kosmos.db.init_from_config()` even for commands that don't need it (version, help). On failure, logs the error and prints a user-friendly message via `print_error()`, but does NOT exit. Database-dependent commands must handle `None` or uninitialized database state on their own.

3. **All-or-nothing command registration**: Commands are imported in a single `from ... import` statement. If any single command module fails to import (e.g., due to a missing dependency), ALL 7 commands disappear because the `ImportError` is caught around the entire import block, not per-command.

4. **`logging.basicConfig` is effectively one-shot**: Called every time `main()` runs, but `basicConfig` is a no-op if handlers already exist on the root logger. If the CLI is invoked multiple times in the same process (e.g., in tests), only the first call's configuration takes effect.

5. **`setup_logging()` side effect when `trace=True`**: Reaches into `kosmos.config.get_config()` and mutates config object fields: `log_llm_calls`, `log_agent_messages`, `log_workflow_transitions`, `stage_tracking_enabled`, `debug_level=3`. Exception is caught and ignored if config is not ready.

**What breaks if you change it**:

- Entry point for all CLI usage: Every `kosmos` command flows through this module's `main()` callback.
- Database initialization on every invocation: The `main()` callback initializes the database even for commands that don't need it.
- Import-time side effects: `load_dotenv()` and `install_rich_traceback()` run when the module is imported, affecting any test that imports it.
- Fragile command registration: A single broken import in `kosmos.cli.commands` disables all 7 dynamically registered commands.

**Runtime Dependencies**:

- `typer` -- CLI framework
- `rich` -- Console, Markdown, Panel, traceback
- `dotenv` -- Environment loading
- Internal: `kosmos.cli.utils` (console, print helpers, get_icon), `kosmos.cli.themes` (KOSMOS_THEME), `kosmos.config`, `kosmos.db`

**Public API**:

- **`setup_logging(verbose, debug, trace, debug_level)`**: Configures Python `logging.basicConfig` with file handler (writing to `kosmos.log` in the log directory) and optionally a `StreamHandler` or `NullHandler`. Side effects: Clears all handlers, creates log directory. Log levels: trace/debug_level>=2 -> DEBUG, debug/debug_level>=1 -> DEBUG, verbose -> INFO, default -> WARNING.
- **`main(ctx, verbose, debug, trace, debug_level, debug_modules, quiet)`**: Typer callback, runs before any subcommand. Stores global options in `ctx.obj` dict. Calls `setup_logging()`, calls `init_from_config()` (creates/migrates database). Quiet mode sets `console.quiet = True`.
- **`version()`**: Displays version string, Python version, platform info, and Anthropic SDK version. Uses Rich Panel.
- **`info()`**: Displays system configuration, cache status, and API key status. Reads filesystem to compute cache size. On config load failure, calls `typer.Exit(1)`.
- **`doctor()`**: Runs 5 diagnostic checks (Python version >= 3.9, 8 required packages, API key, cache directory, database connection + schema validation). Individual check failures do not abort; all checks run. If any fails, exits with code 1.
- **`register_commands()`**: Imports 7 command modules (`run`, `status`, `history`, `cache`, `config`, `profile`, `graph`) and registers them. `ImportError` silently caught.
- **`cli_entrypoint()`**: Installed entry point. Wraps `app()` in try/except. Catches `KeyboardInterrupt` (exit 130) and generic exceptions (print error, exit 1). Debug detection via `"--debug" in sys.argv`.

**Test Coverage**: Three test files exercise this module: `tests/integration/test_cli.py` (417 lines, uses `typer.testing.CliRunner`), `tests/e2e/test_cli_workflows.py` (256 lines), and `tests/unit/cli/test_commands.py` and `tests/unit/cli/test_graph_commands.py`.

**Gotchas**:

1. [HIGH] **`load_dotenv()` at import time**: Importing this module mutates `os.environ` from `.env` file -- test isolation risk.
2. [MEDIUM] **`install_rich_traceback()` at import time**: Global exception handler override on import.
3. [HIGH] **All-or-nothing command registration**: One broken command module import disables all 7 commands silently.
4. [MEDIUM] **Database init on every CLI invocation**: Even `kosmos version` triggers database initialization attempt.
5. [LOW] **`logging.basicConfig` is effectively one-shot**: Repeated calls in the same process are no-ops.
6. [LOW] **Hard-coded version in test**: `test_cli.py:82` asserts `"v0.2.0"` -- will break on version bumps.
7. [LOW] **No `reset` for singleton alert manager or DB state**: Tests importing `app` get shared global state from prior test runs.

---

### `config.py` -- Detailed Behavioral Analysis (see Change Impact Index: Hub 1)

**Module**: `kosmos/config.py`

**What it does**: Central configuration hub for the entire Kosmos system, implemented as a hierarchy of Pydantic `BaseSettings` subclasses that load values from environment variables and `.env` files. Exposes a singleton `KosmosConfig` instance via `get_config()` that is imported by 40 files across the codebase (54 total import occurrences). Every subsystem -- LLM providers, database, Redis, logging, literature APIs, vector DB, Neo4j, safety, performance, monitoring, development, and world model -- has its own typed configuration section.

**Git Risk**: risk=0.96, churn=27, hotfixes=18, authors=4 -- **HIGHEST RISK in the codebase**. Extreme churn with many hotfixes across multiple authors.

**What's non-obvious**:

1. **Multi-provider LLM architecture with backward compatibility.** The `ClaudeConfig` class is aliased as `AnthropicConfig`. The master `KosmosConfig` has three LLM provider fields: `claude` (backward compat), `anthropic` (new name), and `openai`. A `llm_provider` enum field selects which provider is active ("anthropic", "openai", or "litellm"). The `validate_provider_config` model validator enforces that the selected provider's API key is present.

2. **CLI mode detection via all-9s API key.** `ClaudeConfig.is_cli_mode` returns `True` when the API key consists entirely of the digit `9`. This is a convention for bypassing API calls when running under Claude Code CLI. The check `self.api_key.replace('9', '') == ''` means any string of 9s (including "9", "99", "999999999") triggers CLI mode.

3. **Optional provider configs use factory functions.** `_optional_openai_config()`, `_optional_anthropic_config()`, and `_optional_claude_config()` each check `os.getenv()` before instantiating their config class. If the env var is missing, the field is `None`. This means accessing `config.openai.model` will raise `AttributeError` if `OPENAI_API_KEY` was never set.

4. **LiteLLM env var sync is a manual workaround.** The `sync_litellm_env_vars` model validator manually maps `LITELLM_*` environment variables to the nested `LiteLLMConfig` because Pydantic's nested `BaseSettings` models do not automatically pick up parent `.env` file values without `env_nested_delimiter`. It only overrides fields that still have default values.

5. **SQLite path normalization silently converts relative to absolute.** `DatabaseConfig.normalized_url` resolves relative SQLite paths against the project root (`Path(__file__).parent.parent`). This ensures the DB file location is stable regardless of CWD, but the raw `url` field still holds the relative path. Callers must use `normalized_url` to get the absolute version.

6. **The `.env` file path is hardcoded relative to the config module.** Both `LiteLLMConfig` and `KosmosConfig` set `env_file=str(Path(__file__).parent.parent / ".env")`, pointing to the repository root.

**What breaks if you change it**:

Imported by 40 files across the codebase (54 occurrences). Every subsystem depends on `get_config()`. Changes to field names, defaults, or validation rules propagate everywhere. The singleton pattern means any mutation or reload affects all consumers. Key downstream consumers include: `kosmos.core.logging.configure_from_config()`, `kosmos.core.llm.get_client()`, `kosmos.db.__init__`, `kosmos.agents.base`, all CLI commands.

**Runtime Dependencies**:

- `pydantic` and `pydantic_settings` -- external packages
- `kosmos.utils.compat.model_to_dict` -- internal utility
- `os`, `pathlib` (stdlib)

**Public API**:

- **`get_config(reload=False) -> KosmosConfig`**: Returns the global singleton `KosmosConfig`. On first call (or if `reload=True`), instantiates a new `KosmosConfig()` from environment and calls `create_directories()`. Preconditions: Environment variables and/or `.env` file must be populated. Raises `pydantic.ValidationError` if required fields are missing or invalid. Raises `ValueError` from model validators if provider API keys are missing. Side effects: Creates log and ChromaDB directories on disk. Sets module-level `_config` global.
- **`reset_config()`**: Sets the singleton to `None`, forcing re-creation on next `get_config()` call. Never raises.
- **`parse_comma_separated(v)`**: Converts comma-separated string to list. Used as `BeforeValidator` for list fields that may come from env vars.
- **`KosmosConfig.get_active_model() -> str`**: Returns the model string for the currently active LLM provider. Raises `ValueError` for unknown provider.
- **`KosmosConfig.get_active_provider_config() -> dict`**: Returns a dict with `model`, `api_key`, and optionally `api_base` for the active provider. Raises `ValueError` for unknown provider. Raises `AttributeError` if provider config is `None`.
- **`KosmosConfig.validate_dependencies() -> List[str]`**: Returns a list of missing dependency descriptions.
- **`KosmosConfig.to_dict() -> dict`**: Serializes full config to dict using `model_to_dict()` helper.

**Config Section Classes**: `ClaudeConfig`, `OpenAIConfig`, `LiteLLMConfig`, `ResearchConfig`, `DatabaseConfig`, `RedisConfig`, `LoggingConfig`, `LiteratureConfig`, `VectorDBConfig`, `Neo4jConfig`, `SafetyConfig`, `PerformanceConfig`, `LocalModelConfig`, `MonitoringConfig`, `DevelopmentConfig`, `WorldModelConfig`.

**Test Coverage**: `/mnt/c/python/Kosmos/tests/requirements/core/test_req_configuration.py` covers configuration requirements. Config is exercised indirectly by nearly every integration and e2e test via `get_config()`. **Gap**: No direct unit test found for `sync_litellm_env_vars`, `validate_provider_config`, or `DatabaseConfig.normalized_url`.

**Gotchas**:

1. [HIGH] **`config.openai` can be `None`**: Accessing attributes on it without checking raises `AttributeError`. Same for `config.claude` and `config.anthropic` when their API keys are unset.
2. [MEDIUM] **Default model constants are module-level**: `_DEFAULT_CLAUDE_SONNET_MODEL = "claude-sonnet-4-5"` and `_DEFAULT_CLAUDE_HAIKU_MODEL = "claude-haiku-4-5"`. These are imported by `kosmos.models.hypothesis` for `HypothesisGenerationResponse.model_used`. Changing them affects hypothesis metadata.
3. [LOW] **`create_directories()` runs on every `get_config()` call** (first time or reload). Idempotent but touches the filesystem.
4. [LOW] **Neo4j default password is hardcoded**: `default="kosmos-password"`. This should not be used in production.
5. [LOW] **`ResearchConfig.max_runtime_hours`** has upper bound of 24.0 but the description mentions "up to 12 hours continuous operation", creating a discrepancy between documentation and validation bounds.

---

### `core_llm.py` -- Detailed Behavioral Analysis (see Change Impact Index: Hub 5)

**Module**: `kosmos/core/llm.py` (707 lines)

**What it does**: Multi-provider LLM integration hub. Primary gateway through which every agent and code generator obtains LLM access. Contains two generations of client architecture: the legacy `ClaudeClient` class (Anthropic-only, direct API wrapper, lines 108-605) and the newer provider-agnostic `get_client()`/`get_provider()` factory functions (lines 613-706) that delegate to `LLMProvider` subclasses. Thread-safe singleton LLM client with double-checked locking.

**Git Risk**: risk=0.70, churn=12, hotfixes=6, authors=3 -- High risk, frequent changes.

**What's non-obvious**:

1. **Graceful degradation**: `anthropic` imported in try/except. `HAS_ANTHROPIC = False` on failure. Warning printed to stdout. Module importable without SDK; fails at `ClaudeClient()` time.
2. **Two `ClaudeClient` classes**: `llm.py:108` vs `providers/anthropic.py` alias. Import path determines class.
3. **Cache key granularity**: Includes `system`, `max_tokens`, `temperature`, `stop_sequences`. Same prompt with different params misses cache.
4. **`reset_stats()` preserves cache**: Zeroes counters but `ClaudeCache` persists.
5. **20 `COMPLEX_KEYWORDS`**: 'hypothesis', 'experiment', 'algorithm', 'theorem', etc. Biases auto-selection toward Sonnet for research prompts.
6. **CLI mode detection is fragile**: Any string consisting entirely of '9' characters matches, regardless of length.

**What breaks if you change it**:

This module is imported by every agent: `hypothesis_generator.py`, `experiment_designer.py`, `data_analyst.py`, `research_director.py`, plus `code_generator.py` and `domain_router.py` (the only caller using the class directly, not the factory). Any change to the `generate()` signature or return type affects the entire system. The singleton pattern means configuration changes via `get_client(reset=True)` affect all subsequent callers globally.

**Runtime Dependencies**:

- `kosmos.config` for `_DEFAULT_CLAUDE_SONNET_MODEL` and `_DEFAULT_CLAUDE_HAIKU_MODEL`
- `kosmos.core.pricing.get_model_cost` for cost estimation
- `kosmos.core.claude_cache.get_claude_cache` and `ClaudeCache`
- `kosmos.core.utils.json_parser.parse_json_response` and `JSONParseError`
- `kosmos.core.providers.base.LLMProvider` and `ProviderAPIError`
- `kosmos.core.providers.anthropic.AnthropicProvider` (lazy import in fallback path only)
- `kosmos.core.providers.get_provider_from_config` (lazy import in get_client)
- `anthropic` SDK (optional at module level, required for ClaudeClient)
- `threading` for lock-based singleton safety
- `json` for schema serialization in generate_structured

**Public API**:

- **`ClaudeClient.__init__()`**: Preconditions: Requires `anthropic` package (raises `ImportError` if absent). Requires API key from parameter or `ANTHROPIC_API_KEY` env var (raises `ValueError` if missing). Side effects: Initializes `Anthropic` SDK client; optionally initializes `ClaudeCache` singleton. Initializes 8 counter fields to 0 for usage tracking.
- **`ClaudeClient.generate(prompt, system, max_tokens, temperature, stop_sequences, model_override, model_complexity)`**: Single-prompt generation. Model priority: (1) `model_override`, (2) auto-selection via `ModelComplexity`, (3) default. Cache hit returns immediately, increments `cache_hits`. Miss increments `cache_misses`, calls `messages.create()`. Stores response in cache. Mutates counters. `max_tokens` warning logged. Exceptions re-raised.
- **`ClaudeClient.generate_structured(prompt, system, output_schema/schema, max_tokens, temperature, max_retries=2)`**: JSON generation. Appends schema to system prompt. Retries up to `max_retries=2` on `JSONParseError`. Uses `parse_json_response()`. Bypasses cache on retries. Exhausted retries raise `ProviderAPIError(recoverable=False)`. Temperature defaults to 0.3.
- **`ClaudeClient.generate_with_messages(messages, system, max_tokens, temperature)`**: Multi-turn conversation. No caching. No auto model selection, always uses `self.model`. Accepts `List[Dict[str, str]]` messages. Returns `response.content[0].text`. No `stop_sequences` parameter.
- **`ModelComplexity.estimate_complexity(prompt, system)`**: Heuristic 0-100 score. 4 chars/token estimate. Only "haiku" (< 30) or "sonnet" (>= 30) recommendations. Returns dict with score, estimate, keyword matches, recommendation, reason.
- **`get_client(reset=False, use_provider_system=True)`**: Singleton factory. Thread-safe via `_client_lock`. Mutates `_default_client` global. Falls back to `AnthropicProvider` on config failure.
- **`get_usage_stats()`**: Returns dict: request counts, tokens, cost, cache metrics, model selection breakdown. Cost hardcoded to "claude-sonnet-4-5" pricing. Haiku savings uses magic `12 * 0.8` multiplier.

**Test Coverage**: Unit tests at `tests/unit/core/test_llm.py` require real `ANTHROPIC_API_KEY` (not mocked). Tests cover initialization (API/CLI mode), basic generation, structured JSON, multi-turn, usage stats, singleton behavior. CLI mode test sets key to 48 '9' characters. Singleton test verifies identity and reset. No unit tests for `ModelComplexity`, `get_provider()`, or the fallback paths in `get_client()`.

**Gotchas**:

1. [MEDIUM] **`generate_with_messages()` ignores auto model selection**: Always uses `self.model`, not the complexity-based selection. Undocumented behavior.
2. [LOW] **Cost estimation hardcodes "claude-sonnet-4-5"**: Even when other models are used. Savings calculations are rough approximations.
3. [LOW] **CLI mode detection is fragile**: Any string consisting entirely of '9' characters matches, regardless of length.
4. [MEDIUM] **Thread safety covers only initialization**: The singleton is created under lock, but subsequent usage of the client instance is not synchronized.
5. [LOW] **The module imports from `kosmos.core.providers.base` at module level** for both `ProviderAPIError` and `LLMProvider`, creating a dependency chain from the legacy client to the provider system.
6. [LOW] **`generate_structured()` uses lower temperature (0.3)** vs `generate()` default (0.7), which is a deliberate trade-off for more deterministic JSON output.

---

### `core_logging.py` -- Detailed Behavioral Analysis (see Change Impact Index: Hub 4)

**Module**: `kosmos/core/logging.py`

**What it does**: Provides a structured logging system with JSON and human-readable text output, experiment run tracking via `ExperimentLogger`, and a `configure_from_config()` bridge that wires logging to the centralized `KosmosConfig`. The module uses stdlib `logging` exclusively and adds a `contextvars`-based correlation ID for request tracing across async boundaries.

**Git Risk**: risk=0.26, churn=3, hotfixes=1, authors=2 -- Low risk, stable.

**What's non-obvious**:

1. **Almost no modules actually import from this file.** `from kosmos.core.logging import` appears only in its own docstring examples. Instead, 111 files use `import logging; logger = logging.getLogger(__name__)` directly from stdlib. The custom formatters (`JSONFormatter`, `TextFormatter`) only take effect when `setup_logging()` is called early in the application lifecycle -- they attach to the root logger. If `setup_logging()` is never called, all 111 modules log through default stdlib handlers with no JSON structure.

2. **Correlation ID is a ContextVar, not propagated automatically.** The `correlation_id` ContextVar must be explicitly set by calling code; there is no middleware or decorator in this module that sets it. It is only read inside `JSONFormatter.format()`. If no caller ever sets it, all JSON logs will omit `correlation_id`.

3. **Workflow context fields use hasattr checks, not a formal schema.** The `JSONFormatter` checks `hasattr(record, "workflow_id")`, `hasattr(record, "cycle")`, and `hasattr(record, "task_id")`. These extra attributes must be injected by callers via `logging.LogRecord` extra dictionaries. There is no type safety or validation.

4. **`ExperimentLogger.end()` will not raise if `start()` was never called** -- the guard returns `0` silently, masking misuse.

5. **`setup_logging()` mutates global state.** It clears all handlers from the root logger and re-attaches new ones. Calling it twice discards the first configuration entirely.

**What breaks if you change it**:

- 141 connections (per crawl plan). The actual import coupling is indirect: most modules use `import logging` from stdlib, and this module controls behavior only because it configures the root logger. Changing the `JSONFormatter` output schema affects any log-parsing pipeline downstream. Removing `setup_logging()` would break application boot in `configure_from_config()`, which is called from config initialization paths.
- The `ExperimentLogger` is a standalone utility with no imports outside the module. Its blast radius is confined to any code that explicitly instantiates it.

**Runtime Dependencies**:

- `logging`, `logging.handlers` (stdlib)
- `contextvars` (stdlib, Python 3.7+)
- `json`, `sys`, `datetime`, `pathlib`, `enum` (stdlib)
- `kosmos.config.get_config` -- lazy import inside `configure_from_config()`
- No external packages required.

**Public API**:

- **`setup_logging(level, log_format, log_file, debug_mode) -> logging.Logger`**: Configures the root logger with console (stdout) and optional rotating file handler. If `debug_mode=True`, overrides level to DEBUG. Clears ALL existing root logger handlers. Creates log directory if needed. Emits startup log message. Rotating file handler: 10MB max, 5 backups. Error: `OSError` if directory creation fails, `AttributeError` if level invalid.
- **`get_logger(name) -> logging.Logger`**: Thin wrapper around `logging.getLogger(name)`. No side effects. Never raises.
- **`configure_from_config()`**: Reads logging settings from `KosmosConfig` singleton and calls `setup_logging()`. Uses lazy import to avoid circular dependency. Propagates any config loading errors.
- **`class JSONFormatter.format(record) -> str`**: Serializes log record to JSON with timestamp (UTC ISO), level, logger name, message, module, function, line. Includes correlation_id if set, exception info if present, workflow context fields if attached. Uses `datetime.utcfromtimestamp()` which is deprecated in Python 3.12+.
- **`class TextFormatter.format(record) -> str`**: Produces human-readable output with ANSI colors when `use_colors=True` and stdout is a TTY. Mutates `record.levelname` in-place to inject ANSI codes -- this can leak colored level names to other handlers attached to the same logger.
- **`class ExperimentLogger`**: Tracks experiment lifecycle via structured logging events. Accumulates events in `self.events` list. Public methods: `start()`, `log_hypothesis(str)`, `log_experiment_design(dict)`, `log_execution_start()`, `log_result(dict)`, `log_error(str)`, `end(status)`, `get_summary() -> dict`. If `start()` not called before `end()`, duration defaults to `0`. All methods emit log messages via `self.logger`.

**Test Coverage**: Dedicated test file: `/mnt/c/python/Kosmos/tests/requirements/core/test_req_logging.py` covers REQ-LOG-001 through REQ-LOG-006. Tests validate `setup_logging()`, `get_logger()`, JSON timestamp presence, log level configuration. **Gap**: No unit tests found for `ExperimentLogger` class or `configure_from_config()`. No test validates `TextFormatter` ANSI color behavior or the `record.levelname` mutation side effect.

**Gotchas**:

1. [HIGH] **TextFormatter mutates record.levelname**: ANSI escape codes are baked into the level name. If both a `TextFormatter` console handler and a `JSONFormatter` file handler are attached to the same logger, the JSON output may contain ANSI codes in the level field because the record is shared.
2. [MEDIUM] **`datetime.utcfromtimestamp()` deprecation**: This method is deprecated in Python 3.12. Should be replaced with `datetime.fromtimestamp(record.created, tz=timezone.utc)`.
3. [MEDIUM] **ContextVar correlation_id has no auto-propagation**: Must be manually set per request/task. Async task spawning copies context automatically, but thread-pool executors do not unless explicitly wrapped.
4. [LOW] **`ExperimentLogger` has no persistence**: Events are in-memory only. If the process crashes, the experiment log is lost.

---

### `core_workflow.py` -- Detailed Behavioral Analysis (see Change Impact Index: Hub 7)

**Module**: `kosmos/core/workflow.py` (417 lines)

**What it does**: State machine for the autonomous research workflow (Phase 7). Defines the lifecycle states, transitions, and tracking for the iterative research cycle that Kosmos uses to autonomously generate hypotheses, design experiments, execute them, analyze results, and refine. The module is pure state management logic -- no I/O operations, no LLM calls, and no database access.

**Git Risk**: risk=0.45, churn=4, hotfixes=2, authors=3 -- Moderate risk.

**What's non-obvious**:

1. **`mark_supported()` and `mark_rejected()` implicitly call `mark_tested()`.** This means calling `mark_supported()` adds the hypothesis to both `supported_hypotheses` AND `tested_hypotheses`. A caller that manually calls both `mark_supported()` and `mark_tested()` will not create duplicates due to the deduplication check.

2. **CONVERGED is nearly terminal**: Only one exit transition exists (back to GENERATING_HYPOTHESES). There is no transition from CONVERGED to PAUSED or ERROR.

3. **The config import in `transition_to()` is wrapped in try/except**: The `get_config()` call for `log_workflow_transitions` is a lazy import inside a try block. If config loading fails, the workflow silently continues without logging. This means the module works even when config is not initialized.

4. **Transition timestamps use `datetime.utcnow()`**: No timezone awareness. Duration calculations in `get_state_duration()` assume all timestamps are UTC.

5. **No persistence mechanism**: The workflow state exists only in memory. If the process crashes, all transition history is lost. The `ResearchPlan` is a Pydantic model with no save method.

**What breaks if you change it**:

Medium blast radius. Used by the `ResearchDirector` agent to manage the autonomous research loop. The `WorkflowState` enum values propagate to logging, status displays, and database records via `ResearchSession.status`. Integration tests at `tests/integration/test_research_workflow.py` and E2E tests at `tests/e2e/test_full_research_workflow.py` exercise this module through the full stack. The `ResearchPlan` model is the structural backbone of the iteration tracking system -- its `hypothesis_pool`, `tested_hypotheses`, `supported_hypotheses`, and `rejected_hypotheses` lists are the authoritative state for research progress.

**Runtime Dependencies**:

- `pydantic` (BaseModel, Field, ConfigDict)
- `kosmos.config` (lazy import in `transition_to()` only, failure-tolerant)
- Standard library: `enum`, `datetime`, `logging`, `typing`
- No external dependencies, no database access, no LLM access.

**Public API**:

- **`WorkflowState(str, Enum)`**: Nine states: `INITIALIZING`, `GENERATING_HYPOTHESES`, `DESIGNING_EXPERIMENTS`, `EXECUTING`, `ANALYZING`, `REFINING`, `CONVERGED`, `PAUSED`, `ERROR`. All values are lowercase strings.
- **`NextAction(str, Enum)`**: Eight possible next actions: `GENERATE_HYPOTHESIS`, `DESIGN_EXPERIMENT`, `EXECUTE_EXPERIMENT`, `ANALYZE_RESULT`, `REFINE_HYPOTHESIS`, `CONVERGE`, `PAUSE`, `ERROR_RECOVERY`.
- **`WorkflowTransition(BaseModel)`**: Records a single state change. Fields: `from_state`, `to_state`, `action` (string), `timestamp` (default utcnow), `metadata` (dict). Uses `ConfigDict(use_enum_values=True)`.
- **`ResearchPlan(BaseModel)`**: Core fields: `research_question` (required), `domain`, `initial_strategy`, `current_state` (default INITIALIZING). Hypothesis tracking: `hypothesis_pool`, `tested_hypotheses`, `supported_hypotheses`, `rejected_hypotheses` (all lists of IDs). Experiment tracking: `experiment_queue`, `completed_experiments`. Results: `results` list. Iteration: `iteration_count` (starts 0), `max_iterations` (default 10). Convergence: `has_converged`, `convergence_reason`. Mutation methods (all update `updated_at`): `add_hypothesis(id)`, `mark_tested(id)`, `mark_supported(id)`, `mark_rejected(id)`, `add_experiment(id)`, `mark_experiment_complete(id)`, `add_result(id)`, `increment_iteration()`. Query methods: `get_untested_hypotheses()`, `get_testability_rate()`, `get_support_rate()`.
- **`ResearchWorkflow.__init__(initial_state, research_plan)`**: Initializes with default INITIALIZING state and empty `transition_history` list.
- **`ResearchWorkflow.transition_to(new_state, action, metadata)`**: Validates transition against `ALLOWED_TRANSITIONS`, creates `WorkflowTransition` record, updates `current_state`, appends to `transition_history`, synchronizes `research_plan.current_state` if plan exists. Raises `ValueError` if transition not allowed. Optionally logs via config. Returns `True` on success.
- **`ResearchWorkflow.can_transition_to(state)`**: Checks if target state is in the allowed list without transitioning.
- **`ResearchWorkflow.reset()`**: Resets to INITIALIZING, clears `transition_history`. Transition history is lost permanently.
- **`ResearchWorkflow.to_dict()`**: Exports current state, transition count, and last 5 transitions.
- **`ResearchWorkflow.get_state_duration(state)`**: Total seconds spent in a given state by scanning transition history.
- **`ResearchWorkflow.get_state_statistics()`**: Aggregates visit counts and durations for all visited states.

**Transition Table**: INITIALIZING -> GENERATING_HYPOTHESES, PAUSED, ERROR. GENERATING_HYPOTHESES -> DESIGNING_EXPERIMENTS, CONVERGED, PAUSED, ERROR. DESIGNING_EXPERIMENTS -> EXECUTING, GENERATING_HYPOTHESES (backtrack), PAUSED, ERROR. EXECUTING -> ANALYZING, ERROR, PAUSED. ANALYZING -> REFINING, DESIGNING_EXPERIMENTS (immediate retest), PAUSED, ERROR. REFINING -> GENERATING_HYPOTHESES (new cycle), DESIGNING_EXPERIMENTS (follow-up), CONVERGED, PAUSED, ERROR. CONVERGED -> GENERATING_HYPOTHESES (restart). PAUSED -> 6 states (universal resumption). ERROR -> INITIALIZING, GENERATING_HYPOTHESES, PAUSED.

**Test Coverage**: Comprehensive unit tests at `tests/unit/core/test_workflow.py` (469 lines). Test classes: `TestWorkflowTransition`, `TestResearchPlan`, `TestResearchWorkflow`, `TestStateMachineTransitions`. Tests verify deduplication, implicit mark_tested, experiment_queue removal, state duration. **Gap**: No tests for `NextAction` enum usage or config-dependent logging in `transition_to()`.

**Gotchas**:

1. [MEDIUM] **Transition validation prevents skipping states**: Cannot go directly from INITIALIZING to EXECUTING. The research cycle must follow the graph.
2. [MEDIUM] **`reset()` is destructive**: Clears all transition history with no undo.
3. [LOW] **Duration calculations are O(n^2)** in the worst case: `get_state_duration()` does a nested scan of `transition_history`.
4. [LOW] **No maximum history size**: `transition_history` grows unboundedly.

---

### `db.py` -- Detailed Behavioral Analysis

**Module**: `kosmos/db/__init__.py` (203 lines) + `kosmos/db/operations.py` (601 lines)

**What it does**: The database layer for Kosmos, split across two files. `db/__init__.py` handles database engine initialization, session management, and lifecycle operations. `db/operations.py` provides CRUD functions for all entity types (Hypothesis, Experiment, Result, Paper, AgentRecord, ResearchSession) with eager loading, JSON validation, and slow query logging.

**Git Risk**: `db/__init__.py`: risk=0.55, churn=5, hotfixes=3, authors=2. `db/operations.py`: risk=0.22, churn=3, hotfixes=1, authors=1. The init module sees more churn.

**What's non-obvious**:

1. **Immediate commits**: All CRUD functions call `session.commit()`. Cannot batch atomically. `get_session()` also commits on exit, leading to potential double-commits.
2. **Eager loading opt-in**: `get_experiment()` defaults `with_hypothesis=True` while `list_experiments()` defaults `False`. Prevents N+1 queries.
3. **Unconditional table creation**: `create_all()` called every `init_database()`. Idempotent but schema changes need Alembic.
4. **Monkey-patched timing**: `context._query_start_time = time.time()` for slow query logging. Relies on mutable SQLAlchemy context.
5. **Slow query logging parameter name is misleading**: `log_slow_queries(session_factory, threshold_ms)` where `session_factory` is actually the SQLAlchemy engine, not the sessionmaker.
6. **Batch operations bypass slow query monitoring**: The `executemany` guard means batch operations are not monitored for slow queries.

**What breaks if you change it**:

The database layer is consumed by virtually every component that needs persistence. Four test files directly import from the DB layer. The `Base` declarative base from `db/models.py` is shared by all ORM models and referenced by Alembic migrations. The `ResultCollector` calls `create_result()`. Changes to CRUD function signatures or commit behavior could affect any component that writes to the database.

**Runtime Dependencies**:

- `sqlalchemy`: `create_engine`, `event`, `pool` in `__init__.py`; `sessionmaker`, `Session` for session management; `joinedload` and `func` in operations
- `kosmos.db.models`: imports `Base` for table creation; operations imports all 6 ORM models plus 2 enum types
- `kosmos.config.get_config` (lazy import inside `init_from_config()`)
- `kosmos.utils.setup.first_time_setup` (lazy import inside `init_from_config()`)
- `kosmos.db.operations.log_slow_queries` (imported inside `init_database()`)
- Standard library: `logging`, `time`, `contextlib.contextmanager`

**Public API**:

- **`init_database(database_url, echo, pool_size=5, max_overflow=10, pool_timeout=30, enable_slow_query_logging=True, slow_query_threshold_ms=100.0)`**: Creates SQLAlchemy engine with database-type-specific configuration. For SQLite: uses `StaticPool`-like setup with `check_same_thread=False`. For PostgreSQL/MySQL: uses `QueuePool` with `pool_pre_ping=True`. Side effects: Mutates module-level `_engine` and `_SessionLocal` globals. Creates `sessionmaker`. Calls `Base.metadata.create_all()`.
- **`get_session()`**: Context manager yielding a SQLAlchemy `Session`. Raises `RuntimeError` if database not initialized. Auto-commits on clean exit, auto-rolls back on exception, always closes. The rollback itself is not wrapped in a try -- if rollback fails, the original exception is lost.
- **`init_from_config()`**: High-level initialization from Kosmos config. Steps: (1) Loads config, (2) runs `first_time_setup()` which creates `.env` file and runs migrations, (3) calls `init_database()`, (4) verifies schema completeness. If `first_time_setup()` reports errors, they are logged as warnings but do not block initialization.
- **`reset_database()`**: Drops ALL tables and recreates them. Destroys all data. Intended only for testing/evaluation.
- **Hypothesis CRUD**: `create_hypothesis()` validates `related_papers` as list, commits and refreshes. `get_hypothesis()` by ID with optional `joinedload(Hypothesis.experiments)`. `list_hypotheses()` filters by domain/status, orders `created_at DESC`, limit 100. `update_hypothesis_status()` raises `ValueError` if missing.
- **Experiment CRUD**: `create_experiment()` validates `protocol` as required dict. `get_experiment()` with optional eager loading of hypothesis (default True) and results (default False). `list_experiments()` filters by `hypothesis_id`, `status`, `domain`, limit 100. `update_experiment_status()` sets `started_at` on RUNNING, `completed_at` on COMPLETED/FAILED.
- **Result CRUD**: `create_result()` validates `data` (required dict), `statistical_tests` (optional dict), `key_findings` (optional list). `get_result()` by ID. `get_results_for_experiment()` filters by `experiment_id`.
- **Paper CRUD**: `create_paper()` validates `authors` list. `get_paper()` by ID. `search_papers()` filters by domain and `min_relevance`.
- **Agent CRUD**: `create_agent_record()` validates optional `config` dict. `update_agent_record()` sets `stopped_at` when status is "stopped".
- **Research Session CRUD**: `create_research_session()` with `autonomous_mode` default True. `get_research_session()` by ID. `update_research_session()` uses `db_session` parameter name to avoid shadowing `ResearchSession` import.

**Test Coverage**: `tests/unit/db/test_database.py` exists for DB-specific unit tests. E2E tests exercise the database through the full stack. No dedicated unit tests for the validation helpers or slow query logging behavior. No tests for `reset_database()` in isolation.

**Gotchas**:

1. [HIGH] **Immediate commit in CRUD functions**: Cannot batch operations atomically. Each `create_*` or `update_*` calls `session.commit()` and `session.refresh()` independently. The `get_session()` context manager also calls `session.commit()` on clean exit, leading to potential double-commits.
2. [MEDIUM] **`get_session()` rollback can mask exceptions**: The rollback is not wrapped in its own try/except. If `session.rollback()` itself fails (e.g., connection lost), the original exception is lost.
3. [LOW] **SQLite lacks connection pooling**: The `pool_size` parameter has no effect for SQLite databases, but no warning is issued.
4. [LOW] **No soft-delete support**: All models lack a `deleted_at` or `is_deleted` field. No `delete_*` functions are defined in operations.py.
5. [LOW] **Parameter name collision**: `update_research_session` uses `db_session` while all other functions use `session`.
6. [LOW] **Six ORM models imported**: All CRUD functions operate on these models exclusively.

---

### `executor.py` -- Detailed Behavioral Analysis

**Module**: `kosmos/execution/executor.py` (1067 lines)

**What it does**: The code execution engine for Kosmos. Executes LLM-generated Python (and R) code with safety sandboxing, output capture, retry logic with self-correcting code repair, and optional profiling. This is the boundary where AI-generated code meets the host system, making it one of the most security-sensitive modules in the codebase.

**Git Risk**: risk=0.70, churn=12, hotfixes=8, authors=3 -- High risk, heavy churn with many hotfixes.

**What's non-obvious**:

1. The module conditionally imports `DockerSandbox` at lines 25-29; if Docker is unavailable, `SANDBOX_AVAILABLE` is set to `False` and the system falls back silently to restricted execution.
2. `DEFAULT_EXECUTION_TIMEOUT` is hardcoded to 300 seconds, applied to unsandboxed execution only.
3. `SAFE_BUILTINS` at lines 43-83 defines a whitelist of ~70 Python builtins permitted in the restricted exec() environment. Notably includes `print`, all exception classes, and type introspection functions, but excludes `open`, `input`, `getattr`, and `__import__` (replaced with restricted version).
4. `_ALLOWED_MODULES` at lines 86-94 restricts imports to scientific/data libraries (numpy, pandas, scipy, sklearn, etc.) plus safe stdlib modules. The restricted import function checks only the top-level module name.
5. Return value extraction at line 516 looks for `exec_locals.get('results', exec_locals.get('result'))` -- the variable name `results` (plural) takes priority over `result` (singular). This is a convention that executed code must follow.
6. **RetryStrategy handles 10+ error types with pattern-based fixes.** `COMMON_IMPORTS` dict maps 16 common variable names to their import statements. FileNotFoundError returns `None` (no fix), marking the error as terminal to prevent infinite retry loops.

**What breaks if you change it**:

Changes to `SAFE_BUILTINS` or `_ALLOWED_MODULES` directly affect what generated code can do. Modifying retry logic affects all experiment execution. The Windows timeout limitation means long-running code could leak threads. The return value extraction convention (`results`/`result` variable name) is a silent contract -- breaking it means no results are captured.

**Runtime Dependencies**:

- `kosmos.execution.sandbox.DockerSandbox` (optional)
- `kosmos.execution.r_executor.RExecutor` (optional)
- `kosmos.safety.code_validator.CodeValidator` (required for `execute_protocol_code`)
- `kosmos.safety.reproducibility.ReproducibilityManager` (optional, lazy-imported)
- `kosmos.core.profiling.ExecutionProfiler` (optional, lazy-imported)
- `kosmos.utils.compat.model_to_dict`

**Public API**:

- **`ExecutionResult`**: Plain data class. Holds success status, return value, stdout/stderr captures, error info, execution time, optional profile data, and `data_source` indicator ('file' or 'synthetic'). `to_dict()` serializes, with profile data in try/except.
- **`CodeExecutor.__init__(max_retries, retry_delay, allowed_globals, use_sandbox=True, sandbox_config, enable_profiling, profiling_mode, test_determinism, execution_timeout)`**: If `use_sandbox=True` but Docker is unavailable, the flag is silently set to `False`. R executor initialization inherits timeout from sandbox_config or defaults to 300s.
- **`CodeExecutor.execute(code, local_vars, language, llm_client, experiment_context)`**: Main entry point. Auto-detects language (Python vs R). Enters a retry loop that mutates `current_code` across iterations via `RetryStrategy.modify_code_for_retry()`. Determinism checking is optional (runs code a second time with seed=42). Non-determinism produces a warning but does NOT fail. LLM-assisted code repair attempted if `llm_client` is passed. Error behavior: All exceptions caught. Final failure returns `ExecutionResult` with `error_type="MaxRetriesExceeded"`.
- **`CodeExecutor._execute_once(code, local_vars)`**: Routes to sandbox or restricted exec(). Captures stdout/stderr.
- **`CodeExecutor._exec_with_timeout(code, local_vars, timeout)`**: On Unix, uses `signal.SIGALRM`. On Windows, uses `ThreadPoolExecutor` -- the exec'd code runs in a separate thread but `future.result(timeout=...)` raises TimeoutError without killing the thread, so runaway code continues executing.
- **`CodeExecutor.execute_with_data(code, data_path, local_vars)`**: Data path injected both as prepended code assignment AND as a local variable (belt-and-suspenders).
- **`RetryStrategy.modify_code_for_retry(code, error_type, error_message, attempt, llm_client)`**: Attempts LLM repair first (limited to first 2 attempts), then falls through to pattern-based fixes. `should_retry()`: `SyntaxError`, `FileNotFoundError`, and `DataUnavailableError` are explicitly non-retryable.
- **`execute_protocol_code(code, local_vars, use_sandbox, timeout)`**: Convenience function. Code is ALWAYS validated via `CodeValidator` before execution. Validation failure returns immediately without executing.

**Test Coverage**: `tests/unit/execution/test_executor.py` (417 lines). Tests basic execution, error handling, retry logic, output capture, return value extraction, code validation, and sandbox integration (mocked). All unit tests run with `use_sandbox=False`. No integration tests for actual Docker sandbox execution.

**Gotchas**:

1. [HIGH] **Silent sandbox downgrade**: If Docker is unavailable, `use_sandbox` is silently set to `False`. Callers have no callback or exception for this.
2. [HIGH] **Windows timeout leak**: On Windows, `_exec_with_timeout` cannot kill the exec'd thread -- it only stops waiting for it. The thread continues running.
3. [HIGH] **Return value convention**: Executed code MUST set a variable named `results` or `result` for outputs to be captured. Not validated or documented to the code generator.
4. [MEDIUM] **LLM repair prompt injection**: `_repair_with_llm` sends user-generated code directly to an LLM prompt. Adversarial content could manipulate the repair.
5. [MEDIUM] **Retry wrapping recursion**: Most pattern-based fixes wrap code in try/except. If retried again, the already-wrapped code gets wrapped again, producing nested try/except blocks.

---

### `experiment.py` -- Detailed Behavioral Analysis (see Change Impact Index: Hub 6; see Data Contracts)

**Module**: `kosmos/models/experiment.py` (700 lines)

**What it does**: Pydantic data models for experiment design, protocols, and validation. Defines the runtime representation of experiments -- distinct from the SQLAlchemy `Experiment` model in `kosmos.db.models` which handles persistence. Central to the experiment design pipeline, used by the experiment designer agent, execution pipeline, result collector, and code generator. Uses Pydantic v2 features including `field_validator` and `ConfigDict`.

**Git Risk**: risk=0.62, churn=7, hotfixes=4, authors=3 -- Moderate-high risk.

**What's non-obvious**:

1. **Defensive LLM parsing**: Validators use `mode='before'`. Strings to ints: `coerce_sample_size`. Comma strings to lists: `coerce_groups`. Natural language to floats: `parse_effect_size` extracts floats via regex.
2. **Duplicated validators**: `ControlGroup.coerce_sample_size` and `ExperimentProtocol.coerce_protocol_sample_size` are near-identical.
3. **Dual serialization**: `to_dict()` manually converts enums. `use_enum_values=False` means `model_dump()` keeps enum objects.
4. **Cross-module enum**: `ExperimentType` from `kosmos.models.hypothesis`.
5. **`_MAX_SAMPLE_SIZE = 100_000`** serves as a hard upper bound for sample sizes, enforced by validators on both `ControlGroup.sample_size` and `ExperimentProtocol.sample_size`. This prevents LLM-generated protocols from specifying unreasonably large sample sizes.

**What breaks if you change it**:

Imported by 15+ files. `kosmos/agents/experiment_designer.py` (primary consumer, creates ExperimentProtocol instances), `kosmos/execution/result_collector.py` (reads protocol for result collection), `kosmos/execution/code_generator.py` (reads protocol steps and variables), `kosmos/core/memory.py`, integration tests (`test_execution_pipeline.py` imports 7 classes from this module). Any change to `ExperimentProtocol` fields or `to_dict()` output impacts experiment storage, code generation, and result collection.

**Runtime Dependencies**:

- `pydantic` (BaseModel, Field, field_validator, ConfigDict)
- `kosmos.models.hypothesis.ExperimentType`
- `kosmos.config._DEFAULT_CLAUDE_SONNET_MODEL` used only in `ExperimentDesignResponse.model_used` default
- Standard library: `datetime`, `enum`, `logging`, `re` (imported inside `parse_effect_size` validator only)

**Public API**:

- **Enums**: `VariableType` (INDEPENDENT, DEPENDENT, CONTROL, CONFOUNDING), `StatisticalTest` (T_TEST, ANOVA, CHI_SQUARE, CORRELATION, REGRESSION, MANN_WHITNEY, KRUSKAL_WALLIS, WILCOXON, CUSTOM).
- **`Variable(BaseModel)`**: Fields: `name`, `type` (VariableType), `description` (min 10 chars), `values`, `fixed_value`, `unit`, `measurement_method`. Validator enforces non-empty description >= 10 chars.
- **`ControlGroup(BaseModel)`**: Fields: `name`, `description` (min 5), `variables` (Dict), `rationale` (min 10), `sample_size` (optional, ge=1, capped at 100k). `coerce_sample_size` converts string to int, clamps to max, returns None on parse failure.
- **`ProtocolStep(BaseModel)`**: Fields: `step_number` (ge=1), `title` (min 3), `description` (min 10), `action`, `requires_steps`, `requires_resources`, `expected_duration_minutes`, `code_template`, `library_imports`. `ensure_title` replaces empty titles with "Untitled Step".
- **`ResourceRequirements(BaseModel)`**: Compute, cost, time, data, dependency, and parallelization fields. No validation beyond field constraints.
- **`StatisticalTestSpec(BaseModel)`**: Fields: `test_type`, `description` (min 10), `null_hypothesis`, `alternative` ("two-sided"), `alpha` (0.05), `variables`, `groups`, `correction_method`, `required_power` (0.8), `expected_effect_size`.
- **`ExperimentProtocol(BaseModel)`** -- the central model: Fields: `id`, `name` (min 5), `hypothesis_id`, `experiment_type`, `domain`, `description` (min 20), `objective` (min 10), `steps` (min 1), `variables`, `control_groups`, `statistical_tests`, `sample_size` (1-100k), `resource_requirements`, `rigor_score` (0-1), `random_seed`, `generated_by`. Step validation enforces sequential 1..N numbering. Manual `to_dict()` (100 lines). Utilities: `get_step()`, `get_independent_variables()`, `get_dependent_variables()`, `has_control_group()`, `total_duration_estimate_days()`.
- **`ExperimentDesignRequest(BaseModel)`**: Input to experiment designer. Fields: `hypothesis_id`, `preferred_experiment_type`, `max_cost_usd`, `max_duration_days`, `require_control_group` (True), `require_power_analysis` (True), `min_rigor_score` (0.6).
- **`ExperimentDesignResponse(BaseModel)`**: Fields: `protocol`, `hypothesis_id`, `design_time_seconds`, `model_used`, validation results, quality metrics, feasibility assessment. `is_feasible()` checks cost and duration against constraints AND `validation_passed`.
- **`ValidationReport(BaseModel)`**: Fields: `protocol_id`, `rigor_score`, checks, control group, sample size, power analysis, bias detection, reproducibility, severity level, summary, recommendations.

**Test Coverage**: No dedicated unit test file. Tested indirectly through `tests/unit/execution/test_result_collector.py`, `tests/unit/agents/test_experiment_designer.py`, `tests/unit/experiments/test_phase4_basic.py`, integration tests. The LLM-robustness validators lack direct unit tests.

**Gotchas**:

1. [HIGH] **`to_dict()` and `model_dump()` produce different output** due to `use_enum_values=False`. Always use `to_dict()` for serialization that needs string enum values.
2. [MEDIUM] **Step numbering must be a contiguous 1..N sequence**. Gaps will fail validation.
3. [LOW] **`ensure_title` replaces bad titles silently** with "Untitled Step" rather than raising, which can mask LLM output quality issues.
4. [LOW] **ResourceRequirements has no validation beyond field constraints**: No cross-field check that `gpu_memory_gb` is set when `gpu_required=True`.
5. [LOW] **`_experiment_logger` warnings only visible** when logging is configured for the `kosmos.models.experiment` namespace.
6. [LOW] **`created_at` and `updated_at` fields use `datetime.utcnow`** (deprecated in Python 3.12+).

---

### `experiment_cache.py` -- Detailed Behavioral Analysis

**Module**: `kosmos/core/experiment_cache.py` (758 lines)

**What it does**: Persistent caching layer for experiment results with SHA256 fingerprinting for exact match and cosine-similarity-based approximate reuse detection. Uses SQLite for storage. Provides a standalone SQLite-backed experiment cache, separate from the general `CacheManager` system.

**Git Risk**: No git risk entry found -- this module may be newer or less actively changed.

**What's non-obvious**:

1. **Imports `CacheType` and `get_cache_manager` from `kosmos.core.cache_manager` but neither symbol is used anywhere in the module body.** This is either incomplete integration or a refactoring artifact.

2. **No production code imports this module.** The experiment caching is currently accessed only through documentation examples and test infrastructure. Referenced in architectural docs (`docs/developer/architecture.md:349`).

3. **`ExperimentNormalizer.normalize_parameters()` has a subtle list-sorting edge case.** List sorting uses a bare `try/except TypeError`. If a list contains mixed types that are individually sortable but mutually incomparable (e.g., `[1, "a"]`), the original unsorted list is kept silently. This can cause fingerprint divergence for semantically identical experiments.

4. **`cache_result()` uses `INSERT OR REPLACE`** -- repeated caching of the same experiment silently overwrites the previous entry. Increments `total_experiments` stat counter but does NOT decrement on overwrite, so the counter inflates over time.

5. **`find_similar()` is a linear scan.** Loads ALL experiments with non-null embeddings into memory, computes cosine similarity against each. No vector index, no pagination. For large caches, this will degrade.

6. **Each method creates and closes its own `sqlite3.connect()`** -- no connection pooling. For high-throughput scenarios, the repeated connection open/close adds overhead.

7. **`_increment_stat()` opens its own separate connection and does NOT acquire the class lock.** Stat updates can race with other operations that hold the lock and are also writing to the same database.

**What breaks if you change it**:

- Direct: Any component using `get_experiment_cache()` or `ExperimentCache` directly gets persistent SQLite writes in `.kosmos_cache/experiments/`.
- Filesystem: Creates `experiments.db` in the working directory subtree. Multiple processes caching to the same directory can cause SQLite locking contention.

**Runtime Dependencies**:

- `sqlite3`, `hashlib`, `json`, `threading`, `datetime`, `pathlib` (all stdlib)
- `kosmos.core.cache_manager` (imported but unused)

**Public API**:

- **`ExperimentCacheEntry`**: Data container. Fields: `experiment_id`, `hypothesis`, `parameters` (Dict), `results` (Dict), `execution_time` (float), `timestamp` (datetime), `metadata` (Optional Dict), `embedding` (Optional List[float]). `to_dict()` serializes all fields. `from_dict()` reconstructs from dict.
- **`ExperimentNormalizer`**: Static utility. `normalize_parameters(params)` recursively normalizes (rounds floats to 6 decimals, sorts keys/lists). `generate_fingerprint(hypothesis, parameters)` produces SHA256 hash. `extract_searchable_text(hypothesis, parameters)` concatenates hypothesis with scalar parameter values.
- **`ExperimentCache.__init__(cache_dir, similarity_threshold=0.90, enable_similarity=True, max_similar_results=5)`**: Creates cache directory and SQLite database. Thread-safe via `threading.RLock()`.
- **`ExperimentCache.cache_result(hypothesis, parameters, results, execution_time, metadata, embedding) -> str`**: Stores experiment. Generates `experiment_id` as `"exp_{fingerprint[:16]}"`. Uses `INSERT OR REPLACE`. On exception, re-raises (only method that propagates exceptions).
- **`ExperimentCache.get_cached_result(hypothesis, parameters) -> Optional[ExperimentCacheEntry]`**: Exact-match lookup by fingerprint. Returns most recent entry. Increments hit/miss stat. Returns `None` on any exception (silently swallowed).
- **`ExperimentCache.find_similar(hypothesis, parameters, embedding) -> List[Tuple[ExperimentCacheEntry, float]]`**: Similarity search. Returns empty list if `enable_similarity=False` or `embedding=None`. Returns empty list on exception.
- **`ExperimentCache.get_stats() -> Dict`**: Cache statistics including hit count, miss count, hit rate, database file size.
- **`ExperimentCache.clear() -> int`**: Deletes all rows. Destructive with no confirmation.
- **`ExperimentCache.get_recent_experiments(limit=10)`**: Most recent experiments by timestamp.
- **`get_experiment_cache(similarity_threshold, enable_similarity)`**: Singleton accessor. Parameters only applied on first call.
- **`reset_experiment_cache()`**: Sets global to `None`. Does NOT close SQLite connection.

**Database Schema**: Two tables: `experiments` (experiment_id PK, hypothesis, parameters, results, execution_time, timestamp, metadata, embedding, fingerprint, searchable_text, created_at; indexes on fingerprint, timestamp DESC, hypothesis) and `cache_stats` (stat_key PK, stat_value, updated_at; seeded with total_experiments, cache_hits, cache_misses, similar_hits).

**Test Coverage**: **No dedicated test file exists** for `experiment_cache.py`. Zero automated tests.

**Gotchas**:

1. [MEDIUM] **Unused import**: `CacheType` and `get_cache_manager` are imported but never referenced.
2. [HIGH] **No test coverage**: Zero automated tests exist for this 758-line module.
3. [MEDIUM] **`total_experiments` counter inflates**: `INSERT OR REPLACE` increments the stat even on overwrites.
4. [MEDIUM] **Singleton ignores parameter changes**: `get_experiment_cache()` uses first-call parameters, silently ignoring subsequent calls with different thresholds.
5. [LOW] **Linear similarity scan**: `find_similar()` loads all embeddings into memory -- no indexing for scale.
6. [MEDIUM] **`_increment_stat()` bypasses the lock**: Opens its own connection outside the class lock, creating a thread-safety gap.
7. [LOW] **`reset_experiment_cache()` does not close DB**: The old instance's SQLite connection is abandoned (relies on GC).

---

### `guardrails.py` -- Detailed Behavioral Analysis

**Module**: `kosmos/safety/guardrails.py` (453 lines)

**What it does**: Comprehensive safety guardrails for autonomous research execution. Implements emergency stop (via Unix signals and filesystem flag file), resource consumption limits, safety incident logging, and code validation coordination. This is the safety-critical module that prevents runaway or dangerous AI-generated code from executing.

**Git Risk**: risk=0.61, churn=6, hotfixes=5, authors=3 -- Moderate-high risk with frequent hotfixes.

**What's non-obvious**:

1. **`STOP_FLAG_FILE = Path(".kosmos_emergency_stop")` is a RELATIVE path.** The flag file location depends on the current working directory at check time. If the process changes cwd, the flag file check breaks.

2. **`CodeValidator` is initialized with hardcoded permissions**: `allow_file_read=True`, `allow_file_write=False`, `allow_network=False`. These are not configurable through the guardrails constructor, meaning all code validated through guardrails has the same permission profile.

3. **Default resource limits implement deny-by-default security**: `max_cpu_cores=None`, `max_memory_mb=2048`, `max_execution_time=300`, `allow_network_access=False`, `allow_file_write=False`, `allow_subprocess=False`.

4. **Boolean permissions use AND logic**: `requested AND default`. Since defaults are all `False`, requested permissions are always blocked regardless of what's requested. The system cannot grant network, file write, or subprocess access through this interface.

5. **`is_emergency_stop_active()` has a SIDE EFFECT**: If the flag file exists and the stop isn't already active, it TRIGGERS the stop. Merely CHECKING whether emergency stop is active can CAUSE it to become active.

6. **Despite being the central safety coordinator, `SafetyGuardrails` is not currently wired into the execution pipeline.** `CodeExecutor` uses `CodeValidator` directly, and the emergency stop mechanism is not checked during code execution. The guardrails exist but are not enforced in the current execution path.

**What breaks if you change it**:

Changes to emergency stop logic affect the entire system's ability to halt. Changes to resource limit enforcement affect all experiment execution. The flag file mechanism creates a cross-process dependency on the filesystem.

**Runtime Dependencies**:

- `kosmos.models.safety` (Pydantic models: SafetyReport, SafetyIncident, ViolationType, RiskLevel, ResourceLimit, EmergencyStopStatus)
- `kosmos.utils.compat.model_to_dict`
- `kosmos.safety.code_validator.CodeValidator`
- `kosmos.config.get_config`
- `signal` (stdlib, for SIGTERM/SIGINT handlers)
- Docker client (optional, for container killing)

**Public API**:

- **`SafetyGuardrails.__init__(incident_log_path, enable_signal_handlers=True, docker_client)`**: Initializes `CodeValidator` with hardcoded permissions. Default resource limits from config. Registers SIGTERM/SIGINT handlers (silently fails from non-main thread).
- **`SafetyGuardrails.validate_code(code, context) -> SafetyReport`**: Precondition: Emergency stop must NOT be active (raises `RuntimeError` if it is). If validation fails, violations are logged as incidents. Returns SafetyReport.
- **`SafetyGuardrails.enforce_resource_limits(requested_limits) -> ResourceLimit`**: Caps requested limits to defaults using min() for numerics and AND for booleans.
- **`SafetyGuardrails.check_emergency_stop()`**: Dual-source check: reads flag file AND in-memory status. External process can trigger emergency stop by creating `.kosmos_emergency_stop` file. Raises `RuntimeError` if stop is active.
- **`SafetyGuardrails.is_emergency_stop_active()`**: Property-like method with SIDE EFFECT -- if flag file exists, triggers the stop.
- **`SafetyGuardrails.trigger_emergency_stop(source, reason)`**: Sets in-memory status, logs at CRITICAL, creates flag file with JSON content, kills Docker containers with `kosmos.sandbox=true` label, logs SafetyIncident. Flag file creation failure is non-fatal (in-memory still takes effect).
- **`SafetyGuardrails.reset_emergency_stop()`**: Resets in-memory status and removes flag file. If file removal fails, can create desync.
- **`SafetyGuardrails.safety_context(experiment_id)`**: Context manager. Checks emergency stop BEFORE yielding (pre-execution gate). Catches exceptions during execution and checks if they are due to emergency stop. Post-execution check logs a warning only.
- **`SafetyGuardrails.get_recent_incidents(limit, severity)`**: Returns last N incidents. Emergency stop incidents (violation=None) excluded when filtering by severity.
- **`SafetyGuardrails.get_incident_summary()`**: Aggregate statistics. Emergency stop count computed by checking if 'emergency_stop' is in the incident_id string (fragile).

**Test Coverage**: `tests/unit/safety/test_guardrails.py` (652 lines). Comprehensive unit tests covering initialization, code validation, resource limit enforcement, emergency stop (trigger, reset, flag file detection), safety context manager, incident logging. All tests mock `get_config()`. Tests use `enable_signal_handlers=False`. Flag file tests use `os.chdir(tmp_path)`.

**Gotchas**:

1. [HIGH] **Not wired into execution**: `SafetyGuardrails` is not used by `CodeExecutor` or `execute_protocol_code()`. The guardrails exist but are not enforced in the actual execution pipeline.
2. [HIGH] **Read with write side effect**: `is_emergency_stop_active()` can TRIGGER emergency stop, making it unsafe as a pure status check.
3. [MEDIUM] **Relative flag file path**: Depends on cwd. If the process changes directory, flag file checks look in the wrong place.
4. [MEDIUM] **Reset/flag file desync**: If flag file removal fails during `reset_emergency_stop()`, the next status check will re-trigger the stop.
5. [MEDIUM] **Resource limits are purely advisory**: Returns capped limits but does not actually enforce them at the OS level.
6. [LOW] **Signal handler thread restriction**: If instantiated from a non-main thread, signal handler registration silently fails.
7. [LOW] **Docker kill scope**: Emergency stop kills ALL containers with `kosmos.sandbox=true` label, not just those from the current session.

---

### `hypothesis.py` -- Detailed Behavioral Analysis (see Change Impact Index: Hub 3; see Data Contracts)

**Module**: `kosmos/models/hypothesis.py`

**What it does**: Defines the Pydantic data models for the hypothesis lifecycle: `Hypothesis` (core model), `HypothesisGenerationRequest`, `HypothesisGenerationResponse`, `NoveltyReport`, `TestabilityReport`, and `PrioritizedHypothesis`. These are runtime/API models that complement the SQLAlchemy `HypothesisModel` in `kosmos.db.models`. Imported by 27 files across the codebase (30 occurrences), making it a central data contract for the hypothesis pipeline.

**Git Risk**: risk=0.46, churn=5, hotfixes=2, authors=3 -- Moderate risk.

**What's non-obvious**:

1. **Dual model system: Pydantic runtime + SQLAlchemy persistence.** Code throughout the system (e.g., `research_director.py`) must convert between these two representations manually. There is no shared base or automatic conversion.

2. **`statement` validator permits non-predictive hypotheses.** The `validate_statement()` validator checks for predictive words (`will`, `would`, `should`, `increases`, etc.) but only as a soft check -- it passes without raising even when no predictive word is found. The actual enforcement is only: not empty, not a question, and 10-500 characters.

3. **`model_config = ConfigDict(use_enum_values=False)`**: Enum fields like `status` store the enum member, not the string value. The `to_dict()` method explicitly calls `.value` on enums. Code that serializes via `.model_dump()` will get enum objects, not strings, unless `mode="json"` is specified.

4. **Evolution tracking for hypothesis refinement (Phase 7).** Fields `parent_hypothesis_id`, `generation`, `refinement_count`, and `evolution_history` support hypothesis lineage tracking.

5. **Default timestamps use `datetime.utcnow`** which is deprecated in Python 3.12+.

6. **`_DEFAULT_CLAUDE_SONNET_MODEL` imported from config.** The `HypothesisGenerationResponse.model_used` field defaults to this constant. Changing the default model in config changes the default in hypothesis responses.

7. **Testability threshold is surprisingly low.** `is_testable()` defaults to `threshold=0.3`, while `HypothesisGenerationRequest.min_novelty_score` defaults to `0.5`. The default testability bar is lower than the novelty bar.

**What breaks if you change it**:

Imported by 27 files including all hypothesis-related modules, all agent modules, core modules (convergence, feedback, memory), experiment templates (8+), and world model. Changing `Hypothesis` field names or removing fields would break many modules.

**Runtime Dependencies**:

- `pydantic` (BaseModel, Field, field_validator, ConfigDict)
- `datetime`, `enum`, `typing` (stdlib)
- `kosmos.config._DEFAULT_CLAUDE_SONNET_MODEL`
- Minimal dependency footprint. No database, no network, no LLM calls.

**Public API**:

- **`ExperimentType(str, Enum)`**: Values: `COMPUTATIONAL`, `DATA_ANALYSIS`, `LITERATURE_SYNTHESIS`.
- **`HypothesisStatus(str, Enum)`**: Values: `GENERATED`, `UNDER_REVIEW`, `TESTING`, `SUPPORTED`, `REJECTED`, `INCONCLUSIVE`.
- **`Hypothesis(BaseModel)`**: Required: `research_question`, `statement` (10-500 chars), `rationale` (min 20 chars), `domain`. Optional scores: `testability_score`, `novelty_score`, `confidence_score`, `priority_score` (all 0.0-1.0, default None). Evolution fields: `parent_hypothesis_id`, `generation` (1 = original), `refinement_count`, `evolution_history`. `to_dict()`: Manual serialization converting enums and datetimes. `is_testable(threshold=0.3)`: Returns `False` if score is `None`. `is_novel(threshold=0.5)`: Returns `False` if score is `None`. `validate_statement()`: Strips, rejects empty/question, soft-checks for predictive words. `validate_rationale()`: Strips, rejects empty or <20 chars.
- **`HypothesisGenerationRequest(BaseModel)`**: Required: `research_question` (min 10 chars). Defaults: `num_hypotheses=3` (1-10), `max_iterations=1` (1-5), `require_novelty_check=True`, `min_novelty_score=0.5`.
- **`HypothesisGenerationResponse(BaseModel)`**: `get_best_hypothesis()`: Returns highest `priority_score` or first if none scored. `filter_testable(threshold=0.3)`. `filter_novel(threshold=0.5)`.
- **`NoveltyReport(BaseModel)`**: `novelty_score` (0-1), `max_similarity`, `prior_art_detected`, `is_novel`, `novelty_threshold_used=0.75`.
- **`TestabilityReport(BaseModel)`**: `testability_score` (0-1), `is_testable`, `testability_threshold_used=0.3`, `primary_experiment_type`, resource estimates.
- **`PrioritizedHypothesis(BaseModel)`**: Scoring weights: novelty=0.30, feasibility=0.25, impact=0.25, testability=0.20. `update_hypothesis_priority()` mutates the embedded `hypothesis.priority_score` and `hypothesis.updated_at`.

**Test Coverage**: Tested indirectly through many test files (56 files reference hypothesis). Key files: `tests/unit/hypothesis/test_refiner.py`, `tests/unit/hypothesis/test_prioritizer.py`, `tests/unit/hypothesis/test_testability.py`, `tests/unit/hypothesis/test_novelty_checker.py`, `tests/requirements/scientific/test_req_sci_hypothesis.py`. **Gap**: No dedicated `tests/unit/models/test_hypothesis.py` found that tests Pydantic model validation directly.

**Gotchas**:

1. [HIGH] **`to_dict()` and `.model_dump()` produce different output** because `use_enum_values=False`. `to_dict()` converts enums manually; `.model_dump()` returns enum objects by default.
2. [MEDIUM] **`None` scores fail threshold checks**: `is_testable()` and `is_novel()` return `False` when scores are `None`. A newly created hypothesis with no scores will fail both checks.
3. [MEDIUM] **Inconsistent novelty thresholds**: `NoveltyReport.novelty_threshold_used=0.75` vs `HypothesisGenerationRequest.min_novelty_score=0.5` vs `Hypothesis.is_novel(threshold=0.5)`. Three different defaults for "is this novel enough".
4. [LOW] **`datetime.utcnow` deprecation**: Should use `datetime.now(timezone.utc)` for Python 3.12+ compatibility.
5. [LOW] **`get_best_hypothesis()` silently returns the first hypothesis as fallback** when no hypothesis has a `priority_score`.

---

### `knowledge_graph.py` -- Detailed Behavioral Analysis

**Module**: `kosmos/knowledge/graph.py` (1038 lines)

**What it does**: Neo4j-based knowledge graph for scientific literature. Provides full CRUD operations for four node types (Paper, Concept, Method, Author) and five relationship types (CITES, USES_METHOD, DISCUSSES, AUTHORED, RELATED_TO). This is the persistence backbone for Kosmos's scientific knowledge, wrapping the `py2neo` library.

**Git Risk**: risk=0.24, churn=2, hotfixes=1, authors=2 -- Low risk, stable.

**What's non-obvious**:

1. The class uses `py2neo` library (community library), not the official Neo4j Python driver. `Neo4jError` is imported from `py2neo.errors` but is never used -- dead import.

2. **Constructor can trigger Docker operations.** If `auto_start_container=True`, the constructor calls `_ensure_container_running()` which shells out to `docker` and `docker-compose` to start a Neo4j container.

3. **Connection failure is caught silently.** `self.graph` is set to `None` and `self._connected` to `False`. The constructor does NOT raise. Callers must check `self.connected` before using the graph.

4. **Health check loop uses hardcoded credentials.** Retries up to 30 times with 2-second sleep (max 60 seconds). Shells out to `docker exec` running `cypher-shell` with hardcoded `neo4j`/`kosmos-password`.

5. **`get_paper()` performs a cascade of 4 separate queries** to find a paper: by primary id, by DOI, by arxiv_id, by pubmed_id. This is a performance concern for frequently-called lookups.

6. **Counter increment bug in three relationship creation methods.** `create_authored`, `create_discusses`, and `create_uses_method` increment counters (paper_count, frequency, usage_count) on every call, even when the relationship already exists via merge. Repeated calls produce incorrect counts.

7. **`clear_graph()` runs `MATCH (n) DETACH DELETE n`** -- deletes ALL data in the database. No confirmation, no undo.

**What breaks if you change it**:

This module is the sole persistence layer for Kosmos's knowledge graph. All scientific knowledge flows through here. Cross-module callers: `world_model/simple.py` (primary consumer), `knowledge/__init__.py` (re-exports), `knowledge/graph_builder.py`, `knowledge/graph_visualizer.py`, `agents/literature_analyzer.py`, `literature/citations.py`. Changes to node schemas, relationship types, or query patterns affect every downstream consumer. The singleton pattern means connection issues affect the entire application.

**Runtime Dependencies**:

- `py2neo` (external, required)
- `kosmos.config.get_config` (for connection settings)
- `kosmos.literature.base_client.PaperMetadata` (data model)
- Docker + docker-compose (optional, for auto-start)

**Public API**:

- **`KnowledgeGraph.__init__(uri, user, password, database, auto_start_container=True, create_indexes=True)`**: Connection verification via `RETURN 1` Cypher query. Creates 9 database indexes. Side effects: May start Docker container, network connection.
- **Paper CRUD**: `create_paper(paper: PaperMetadata, merge=True)` performs read-then-write. `get_paper(paper_id)` cascades 4 queries. `update_paper(paper)` delegates to get_paper then push. `delete_paper(paper_id)` cascading delete (removes node AND all relationships).
- **Author CRUD**: Same merge-or-create pattern. Authors identified by `name` (string) -- fragile for disambiguation.
- **Concept CRUD**: New concepts start with `frequency: 0`.
- **Method CRUD**: New methods start with `usage_count: 0`.
- **Relationship Creation**: `create_citation()` checks both papers exist. `create_authored()` increments `paper_count` unconditionally. `create_discusses()` increments `frequency` unconditionally. `create_uses_method()` increments `usage_count` unconditionally. `create_related_to()` bidirectional similarity, no counter.
- **Graph Queries**: `get_citations(paper_id, depth)`, `get_citing_papers(paper_id)`, `find_related_papers(paper_id, max_hops)`, `get_concept_cooccurrence()`.
- **Statistics and Cleanup**: `get_stats()` runs 9 separate Cypher queries. `clear_graph()` deletes ALL data.
- **Singleton**: `get_knowledge_graph()` lazily creates. `reset_knowledge_graph()` sets to None. NOT thread-safe.

**Test Coverage**: `tests/unit/knowledge/test_graph.py` (285 lines). ALL tests require a running Neo4j instance (`pytestmark = pytest.mark.requires_neo4j`). Uses UUID-based test isolation. No mock-based unit tests -- tests cannot run without Neo4j.

**Gotchas**:

1. [HIGH] **Counter increment bug**: `create_authored`, `create_discusses`, and `create_uses_method` increment counters on every call, even when the relationship already exists.
2. [MEDIUM] **4-query paper lookup**: `get_paper()` tries 4 separate queries sequentially.
3. [LOW] **Hardcoded Docker credentials**: Health check uses `kosmos-password`, not from config.
4. [MEDIUM] **Silent connection failure**: Constructor catches all errors and continues. Code that does not check `.connected` will get `AttributeError` on `None` graph.
5. [LOW] **Non-thread-safe singleton**: Race condition on first access.
6. [LOW] **Cascading delete**: All delete methods remove ALL relationships with no confirmation.

---

### `plan_reviewer.py` -- Detailed Behavioral Analysis (see Change Impact Index: Hub 10)

**Module**: `kosmos/orchestration/plan_reviewer.py` (390 lines)

**What it does**: Validates research plans on 5 quality dimensions before execution. Acts as a gate between plan creation and task delegation in the orchestration pipeline. Sits between the `PlanCreatorAgent` and the `DelegationManager` in the research loop. After plan creation and novelty detection, `review_plan()` is called. If rejected, the plan creator gets one revision attempt. If still rejected, execution is skipped entirely.

**Git Risk**: risk=0.45, churn=4, hotfixes=2, authors=3 -- Moderate risk.

**What's non-obvious**:

1. **`DIMENSION_WEIGHTS` are dead code.** The class variable defines weights for 5 dimensions (specificity 0.25, relevance 0.25, novelty 0.20, coverage 0.15, feasibility 0.15) but the docstring explicitly states "not currently used, but available for future". Scoring uses unweighted average.

2. **`review_plan()` is synchronous despite tests marking it `@pytest.mark.asyncio`** and using `await` on it. This is a sync/async contract mismatch.

3. **On LLM parse failure, default scores are 5.0** -- borderline scores that almost always reject (because `min_average_score=7.0`). A malformed LLM response creates silent quality degradation rather than an explicit error signal.

4. **LLM failure falls back to mock review.** Exceptions during LLM review cause a fallback to `_mock_review()`. The mock always approves structurally valid plans (avg=7.5) and always rejects invalid ones (avg=6.0). No way to distinguish "LLM was down" from "LLM approved" without checking the `feedback` field for the "Mock review" string.

5. **Structural requirements are weaker than documented.** The module docstring claims `required_skills` is a required field, but `_meets_structural_requirements` only checks `description` and `expected_output`.

**What breaks if you change it**:

- Blocking gate for all research plan execution. A bug in structural validation or score calculation could silently approve bad plans or reject all plans.
- Mock fallback means LLM outages do not block research -- but plans get rubber-stamped with 7.5 scores, bypassing quality control.
- No persistent state. Results flow through the research loop pipeline only.

**Runtime Dependencies**:

- `anthropic` client (optional, enables LLM-based review)
- `kosmos.config._DEFAULT_CLAUDE_SONNET_MODEL`
- Standard library: `json`, `logging`

**Public API**:

- **`PlanReview` (dataclass)**: Fields: `approved` (bool), `scores` (Dict[str, float]), `average_score`, `min_score`, `feedback` (str), `required_changes` (List[str]), `suggestions` (List[str]). `to_dict()` serializes.
- **`PlanReviewerAgent.__init__(anthropic_client, model, min_average_score=7.0, min_dimension_score=5.0, temperature=0.3)`**: Client can be `None` for mock mode.
- **`PlanReviewerAgent.review_plan(plan: Dict, context: Dict) -> PlanReview`**: Synchronous. Dual-path: if client is None, delegates to `_mock_review()`. If client exists, calls `client.messages.create()` with `max_tokens=2000`. Approval criteria (three-part AND): average >= `min_average_score`, minimum dimension >= `min_dimension_score`, structural requirements pass. Error: catches all exceptions, falls back to mock.
- **`PlanReviewerAgent.get_approval_statistics(reviews: List[PlanReview]) -> Dict`**: Batch statistics over reviews. Pure computation.
- **`_build_review_prompt(plan, context)`**: Embeds full plan JSON in prompt. For large plans, this consumes significant tokens.
- **`_parse_review_response(response_text)`**: Extracts JSON via `find('{')`/`rfind('}')` heuristic. Scores clamped to [0, 10]. Parse failure returns defaults (5.0 all dimensions).
- **`_meets_structural_requirements(plan)`**: Three checks: at least 3 `data_analysis` tasks, at least 2 distinct task types, every task must have `description` and `expected_output`.
- **`_mock_review(plan, context)`**: Returns optimistic scores (base 7.5 if structural requirements pass, 6.0 otherwise). Always appends suggestion noting it is a mock review.

**Test Coverage**: `tests/unit/orchestration/test_plan_reviewer.py` (522 lines). Tests cover dataclass creation, initialization, structural requirements (6 cases), mock review, LLM review with mocked client, prompt building, JSON parsing, approval statistics, edge cases. Integration test references in `tests/integration/test_orchestration_flow.py` and `tests/unit/workflow/test_research_loop.py`.

**Gotchas**:

1. [MEDIUM] **DIMENSION_WEIGHTS are dead code**: Scoring uses unweighted average despite weights being defined.
2. [MEDIUM] **Sync method tested with async**: `review_plan()` is synchronous but test fixtures use `@pytest.mark.asyncio` and `await`.
3. [MEDIUM] **Docstring claims `required_skills` is checked**: Implementation only checks `description` and `expected_output`.
4. [MEDIUM] **LLM parse failure defaults to 5.0**: Borderline scores that almost always reject, creating silent quality degradation.
5. [LOW] **Mock fallback always approves structurally valid plans**: No way to distinguish LLM down from LLM approved without inspecting feedback field.

---

### `kosmos/core/providers/base.py` -- Detailed Behavioral Analysis (see Change Impact Index: Hub 5)

**Module**: `kosmos/core/providers/base.py` (485 lines)

**What it does**: Defines the entire provider abstraction layer for Kosmos's multi-LLM support. Contains four exported types -- three dataclasses (`Message`, `UsageStats`, `LLMResponse`) and one abstract base class (`LLMProvider`) -- plus a custom exception (`ProviderAPIError`). Every LLM interaction in Kosmos passes through the interfaces defined here. Imported by 56 files across the codebase.

**Git Risk**: risk=0.35, churn=3, hotfixes=2, authors=1 -- Low-moderate risk, single author.

**What's non-obvious**:

1. **`LLMResponse` implements 25 string-delegation methods** (`__str__`, `__contains__`, `__len__`, `__add__`, `strip`, `split`, `find`, `replace`, etc.) This string-compatibility shim exists because consumer code historically received raw strings from LLM calls. Callers like `hypothesis_generator.py:282` call `response.strip().lower()` and `code_generator.py:912` calls `response.find("```python")`.

2. **String-compat methods return `str`, not `LLMResponse`.** So `response.strip()` returns a plain string, losing `usage`, `model`, and `metadata`. Any chaining that expects to recover metadata after a string operation will fail silently. Also, `isinstance(response, str)` returns `False`.

3. **`__iter__` iterates over individual characters of `content`**, not over lines or tokens. A `for chunk in response` loop yields single chars.

4. **`total_tokens` is NOT auto-computed** from `input_tokens + output_tokens`. Each caller manually computes it. The `generate_async` methods in anthropic.py and openai.py construct `UsageStats` with only `input_tokens` and `output_tokens`, leaving `total_tokens` at its default of 0.

5. **Usage counters on `LLMProvider` are not thread-safe.** `request_count += 1` is not atomic. In concurrent scenarios (Kosmos uses async LLM calls), counts could be inaccurate.

6. **`generate_stream_async` has dead code.** The `raise NotImplementedError` means the `if False: yield` is unreachable -- it exists solely to make Python's parser classify the function as an async generator.

7. **`ProviderAPIError.is_recoverable()` has ordering bias.** Recoverable patterns are checked before non-recoverable. A message matching both defaults to recoverable. This is a deliberate retry-favoring design.

**What breaks if you change it**:

- `LLMResponse` field changes would break all consumer code in `kosmos/agents/`, `kosmos/execution/`, `kosmos/analysis/`, `kosmos/hypothesis/`, `kosmos/core/domain_router.py`.
- `LLMProvider` method signature changes would require updates to all 3 concrete providers plus the legacy `ClaudeClient`.
- `ProviderAPIError` changes would affect `async_llm.py` (CircuitBreaker), `delegation.py` (error recovery), and 6+ test files.
- `Message` changes would break `generate_with_messages` in all providers.

**Runtime Dependencies**:

- Standard library only: `dataclasses`, `typing`, `abc`, `datetime`, `json`, `logging`
- No external packages. Pure interface module.

**Public API**:

- **`Message` (dataclass)**: Fields: `role` (bare str, not enum), `content`, optional `name`, optional `metadata` dict.
- **`UsageStats` (dataclass)**: `input_tokens`, `output_tokens`, `total_tokens` (NOT auto-computed), optional `cost_usd`, `model`, `provider`, `timestamp`.
- **`LLMResponse` (dataclass)**: `content` (str), `usage` (UsageStats), `model` (str), optional `finish_reason`, `raw_response`, `metadata`. Plus 25 string-delegation methods that return `str`.
- **`LLMProvider(ABC).__init__(config: Dict)`**: Derives `provider_name` by stripping "Provider" and lowercasing. Initializes 4 mutable usage counters (not thread-safe).
- **Abstract methods**: `generate(prompt, system?, max_tokens=4096, temperature=0.7, ...) -> LLMResponse`, `generate_async(same) -> LLMResponse`, `generate_with_messages(messages: List[Message], ...) -> LLMResponse`, `generate_structured(prompt, schema: Dict, ...) -> Dict[str, Any]` (note: returns Dict, not LLMResponse).
- **Concrete methods**: `generate_stream()` (raises NotImplementedError by default), `generate_stream_async()` (raises NotImplementedError with dead `yield`), `get_usage_stats()`, `_update_usage_stats()`, `reset_usage_stats()`, `get_model_info()` (abstract).
- **`ProviderAPIError`**: Fields: `provider`, `message`, `status_code`, `raw_error`, `recoverable` (default True). `is_recoverable()` two-phase: status code (4xx except 429 non-recoverable), then message pattern (9 recoverable patterns checked before 10 non-recoverable).

**Concrete implementations**: AnthropicProvider (882 lines, CLI mode, caching, auto model selection, streaming), OpenAIProvider (644 lines, provider type detection from base_url, token estimation for local models, no streaming), LiteLLMProvider (593 lines, lazy import, Qwen special handling, streaming, own JSON cleaning). Factory: `_PROVIDER_REGISTRY` with aliases (claude -> anthropic, ollama/deepseek/lmstudio -> LiteLLM).

**Test Coverage**: **No unit tests for base.py itself.** Testing gaps: no tests for `LLMResponse` string-compat methods, `ProviderAPIError.is_recoverable()` logic, `UsageStats` default behaviors, `LLMProvider._update_usage_stats` accumulation. The only provider test (`test_litellm_provider.py`) requires live API keys.

**Gotchas**:

1. [HIGH] **`LLMResponse` is not `str`**: `isinstance(response, str)` fails. Code must use duck typing or access `.content`.
2. [HIGH] **String methods return `str`, not `LLMResponse`**: Metadata is lost after any string operation.
3. [MEDIUM] **`__iter__` yields characters**: `for x in response` gives single chars, not lines or tokens.
4. [MEDIUM] **`total_tokens` not auto-computed**: Some async paths leave it at 0.
5. [MEDIUM] **Usage counters not thread-safe**: Concurrent calls can produce inaccurate totals.
6. [LOW] **`generate_stream_async` has dead code**: The `if False: yield` after `raise` is unreachable.
7. [LOW] **`is_recoverable()` has ordering bias**: Ambiguous messages default to recoverable.
8. [HIGH] **No unit tests for base types**: All testing relies on integration/manual tests with live API keys.
9. [MEDIUM] **`generate_structured` returns `Dict`, not `LLMResponse`**: Usage stats from structured calls only available via `get_usage_stats()`.
10. [LOW] **Qwen special-casing only in LiteLLM**: Running Qwen through OpenAI-compatible endpoint skips the no-think directive.

---

### `research_director.py` -- Detailed Behavioral Analysis (see Change Impact Index: Hub 2)

**Module**: `kosmos/agents/research_director.py` (2953 lines)

**What it does**: The master orchestrator agent for autonomous scientific research (Phase 7). Coordinates the full research cycle -- hypothesis generation, experiment design, code execution, result analysis, hypothesis refinement, and convergence detection -- using both message-based async coordination and direct agent calls. At 2953 lines, it is the most complex agent in the system and the central hub through which all research workflow state flows.

**Git Risk**: risk=0.83, churn=21, hotfixes=17, authors=3 -- **VERY HIGH RISK**. Second highest risk score. Extremely high churn and hotfix count.

**What's non-obvious**:

1. **Dual communication patterns: message-based AND direct calls.** The module contains both async `_send_to_*` methods (message-based) and `_handle_*_action()` methods (direct calls). The direct-call pattern was introduced because ConvergenceDetector and other "agents" are actually utility classes not registered in the message router, causing `send_message()` to silently fail. The `_send_to_convergence_detector()` method is explicitly marked DEPRECATED and now delegates to the direct handler.

2. **Two separate locking systems coexist.** Asyncio locks for async code and threading locks for sync contexts. The async-compatible context managers are actually no-op (yield without locking), providing no actual thread safety for async code paths.

3. **Infinite loop prevention via action counter.** `MAX_ACTIONS_PER_ITERATION = 50` and `_actions_this_iteration` counter force convergence if exceeded (Issue #51 fix).

4. **World model integration is entirely optional.** If `get_world_model()` fails, `self.wm = None` and all `_persist_*_to_graph()` methods early-return.

5. **Lazy agent initialization pattern.** Sub-agents initialized to `None` and only instantiated on first use in handler methods, avoiding circular imports and expensive initialization.

6. **Error recovery uses synchronous `time.sleep()` in async context.** This blocks the event loop during backoff. Should use `await asyncio.sleep()`.

7. **Database initialization is attempted in `__init__`.** The constructor calls `init_from_config()` and swallows `RuntimeError` if the DB is already initialized, creating a race condition if multiple instances are created concurrently.

8. **Multiple comparison correction (FDR).** Applies Benjamini-Hochberg FDR correction to p-values before convergence checks.

**What breaks if you change it**:

Referenced by 5 files. It is the entry point for all research workflows. Changes to `decide_next_action()`, `execute()`, or `process_message()` affect the entire research pipeline. Knowledge graph persistence failures are silently swallowed -- graph data loss is invisible at runtime.

**Runtime Dependencies**:

Heavy dependency fan-out:
- `kosmos.agents.base` (BaseAgent, AgentMessage, MessageType, AgentStatus)
- `kosmos.utils.compat.model_to_dict`
- `kosmos.core.rollout_tracker.RolloutTracker`
- `kosmos.core.workflow` (ResearchWorkflow, ResearchPlan, WorkflowState, NextAction)
- `kosmos.core.convergence` (ConvergenceDetector, StoppingDecision, StoppingReason)
- `kosmos.core.llm.get_client`
- `kosmos.core.stage_tracker.get_stage_tracker`
- `kosmos.models.hypothesis` (Hypothesis, HypothesisStatus)
- `kosmos.world_model` (get_world_model, Entity, Relationship)
- `kosmos.db` (get_session)
- `kosmos.db.operations` (get_hypothesis, get_experiment, get_result)
- `kosmos.agents.skill_loader.SkillLoader`
- Lazy imports: hypothesis_generator, experiment_designer, code_generator, executor, data_analyst, refiner, async_llm, parallel

**Public API**:

- **`__init__(research_question, domain, agent_id, config)`**: Initializes director with research question, creates workflow state machine, convergence detector, LLM client, and optional world model. Side effects: DB init, world model entity creation.
- **`async execute(task) -> dict`**: Entry point for research execution. Supports "start_research" (generates plan, starts workflow) and "step" (single research step).
- **`execute_sync(task) -> dict`**: Synchronous wrapper. Tries running event loop, falls back to `asyncio.run()`.
- **`decide_next_action() -> NextAction`**: Core decision engine. Checks budget, runtime limits, loop guard, then uses state-based decision tree. Defaults to GENERATE_HYPOTHESIS for unknown states.
- **`process_message(message: AgentMessage)`**: Routes incoming messages by `agent_type` metadata. Non-async override of async base method.
- **`generate_research_plan() -> str`**: Uses LLM to generate initial plan. Returns error string on failure.
- **`get_research_status() -> dict`**: Comprehensive status snapshot.
- **`execute_experiments_batch(protocol_ids) -> List[dict]`**: Parallel execution when concurrent mode enabled. Falls back to sequential with deprecated `asyncio.get_event_loop().run_until_complete()`.
- **`async evaluate_hypotheses_concurrently(hypothesis_ids)`**: Concurrent evaluation via `AsyncClaudeClient.batch_generate()`.
- **`async analyze_results_concurrently(result_ids)`**: Concurrent analysis.
- **Handler methods**: `_handle_generate_hypothesis_action()`, `_handle_design_experiment_action()`, `_handle_execute_experiment_action()`, `_handle_analyze_result_action()`, `_handle_refine_hypothesis_action()`, `_handle_convergence_action()`.
- **`_handle_error_with_recovery(error_source, error_message, recoverable, error_details)`**: Exponential backoff (2, 4, 8 seconds), circuit breaker after 3 consecutive errors.

**Test Coverage**: Unit tests at `tests/unit/agents/test_research_director.py` (requires `ANTHROPIC_API_KEY` for some tests). Loop prevention tests at `tests/unit/agents/test_research_director_loops.py`. Integration tests at `tests/integration/test_async_message_passing.py`. **Gap**: No tests for direct-call handler methods, concurrent operations, or the FDR correction logic.

**Gotchas**:

1. [HIGH] **`time.sleep()` in async path**: `_handle_error_with_recovery()` uses blocking sleep, halting the event loop during backoff.
2. [MEDIUM] **`asyncio.get_event_loop().run_until_complete()`**: Deprecated pattern in `execute_experiments_batch()` fallback. Will warn or fail in Python 3.12+.
3. [LOW] **`_json_safe()` helper defined inside a method**: Nested function redefined on every call. Minor performance cost.
4. [HIGH] **`_workflow_context()` yields without locking**: The async-compatible version is a no-op context manager, providing no actual thread safety.
5. [MEDIUM] **Knowledge graph persistence failures are silently swallowed**: All `_persist_*_to_graph()` methods catch all exceptions. Graph data loss is invisible at runtime.

---

### `result.py` -- Detailed Behavioral Analysis (see Change Impact Index: Hub 8; see Data Contracts)

**Module**: `kosmos/models/result.py` (378 lines)

**What it does**: Pydantic data models for experiment results, statistical analysis, and result export. Defines the structured representation of what comes back from experiment execution. Complements the SQLAlchemy `Result` model in `kosmos.db.models` -- this module handles runtime validation while the DB model handles persistence.

**Git Risk**: risk=0.59, churn=5, hotfixes=4, authors=3 -- Moderate risk.

**What's non-obvious**:

1. **Dual serialization approach**: `to_dict()` uses the `kosmos.utils.compat.model_to_dict` helper (Pydantic v1/v2 compatible), while `to_json()` uses the Pydantic v2-native `model_dump_json()`. These two methods may produce slightly different output for edge cases.

2. **Version tracking supports re-runs**: The `version` field (default 1) and `parent_result_id` enable creating new versions of results. The `ResultCollector.create_version()` method uses this to track experiment re-runs.

3. **The `validate_primary_test` validator handles raw dicts**: In Pydantic v2, `info.data` contains raw (not yet validated) data, so the validator checks for both dict and model instances.

4. **`supports_hypothesis` is a tri-state**: `True`, `False`, or `None` (no determination made). The `ResultCollector` sets this to None when no p-value is available.

5. **Both `FAILED` and `ERROR` exist in `ResultStatus`**. `FAILED` implies the experiment ran but did not produce useful results, while `ERROR` implies a runtime/system error prevented execution.

**What breaks if you change it**:

Consumed by `kosmos/execution/result_collector.py` (primary producer), `kosmos/agents/data_analyst.py` (reads results), `kosmos/agents/research_director.py` (reads `supports_hypothesis` for iteration decisions), `ResultExport` for CLI report generation. Changes to `ExperimentResult` fields affect result storage, analysis, and the autonomous research loop's convergence logic. The `supports_hypothesis` field is particularly critical -- it drives hypothesis support/rejection decisions.

**Runtime Dependencies**:

- `pydantic` (BaseModel, Field, field_validator)
- `kosmos.utils.compat` (for `model_to_dict`)
- `pandas` (lazy import in `ResultExport.export_csv()` only)
- Standard library: `datetime`, `enum`, `json`, `typing`

**Public API**:

- **`ResultStatus(str, Enum)`**: SUCCESS, FAILED, PARTIAL, TIMEOUT, ERROR.
- **`ExecutionMetadata(BaseModel)`**: Timing (start/end/duration), system (python_version, platform, hostname), resources (cpu_time, memory_peak), environment (random_seed, library_versions), IDs (experiment_id, protocol_id, hypothesis_id), execution details (sandbox_used, timeout_occurred, errors, warnings, data_source).
- **`StatisticalTestResult(BaseModel)`**: test_type, test_name, statistic, p_value (0-1), effect_size, confidence_interval, three significance booleans (0.05, 0.01, 0.001), significance_label, is_primary, sample_size, degrees_of_freedom, additional_stats, interpretation. All three significance fields are required (no defaults).
- **`VariableResult(BaseModel)`**: variable_name, variable_type, summary stats (mean, median, std, min, max), values list, n_samples, n_missing.
- **`ExperimentResult(BaseModel)`** -- central model: Required: experiment_id, protocol_id, status, metadata. Optional: id, hypothesis_id, raw_data, processed_data, variable_results, statistical_tests, primary statistics, version/parent_result_id, stdout/stderr/generated_files, interpretation fields, timestamps. Validators: `validate_statistical_tests` (duplicate test_name check), `validate_primary_test` (named test must exist). Methods: `get_primary_test_result()`, `to_dict()`, `to_json()`, `from_dict()`, `from_json()`, `is_significant(alpha=0.05)`, `get_summary_stats()`.
- **`ResultExport(BaseModel)`**: `export_csv()` (lazy pandas import), `export_markdown()` (structured report with headers).

**Test Coverage**: Tested via `tests/unit/execution/test_result_collector.py` (460 lines). Covers result collection for success/failure/timeout/partial, statistical test creation, variable results, hypothesis support determination, result export, result versioning, database storage (mocked). Significance labels tested: "***"/"**"/"*"/"ns". Does not directly test model validators.

**Gotchas**:

1. [HIGH] **`export_markdown()` will crash on None stats**: The `:.2f` format strings in the variable statistics table do not handle None values. Latent bug for incomplete variable results.
2. [LOW] **No cascading relationship to ExperimentProtocol**: The Pydantic model stores only `protocol_id` as a string. Separate lookup needed.
3. [MEDIUM] **`statistical_tests` requires unique test_names**: Duplicate names cause validation failure.
4. [LOW] **pandas is a hidden dependency**: `export_csv()` fails at runtime if pandas is not installed.

---

### `world_model_simple.py` -- Detailed Behavioral Analysis

**Module**: `kosmos/world_model/simple.py` (1160 lines)

**What it does**: Neo4j-based world model implementation (Simple Mode). Implements the `WorldModelStorage` and `EntityManager` abstract interfaces by wrapping the existing `KnowledgeGraph` class using the Adapter pattern. This is the primary world model backend used by 90% of researchers (per the module's own documentation). Single class `Neo4jWorldModel` inherits from both `WorldModelStorage` and `EntityManager` (multiple inheritance).

**Git Risk**: risk=0.55, churn=5, hotfixes=3, authors=2 -- Moderate risk.

**What's non-obvious**:

1. **Type-dispatch pattern throughout**: Standard entity types (Paper, Concept, Author, Method) are routed to existing KnowledgeGraph methods, while custom types fall through to generic Cypher queries.

2. **If Neo4j is not connected, methods return silently without persisting.** `add_entity()` returns `entity.id` silently. The caller has no indication that the entity was NOT stored.

3. **Parameter mismatch bugs exist in at least three code paths**: `_add_author_entity` and `_add_method_entity` pass a `metadata` kwarg that `KnowledgeGraph` methods don't accept. `add_relationship` for CITES passes `paper_id` instead of `citing_paper_id`. These would raise `TypeError` at runtime.

4. **Cypher injection vectors**: Entity type, relationship type, and project name are injected via f-strings into Cypher queries without sanitization. `entity.type` is directly placed in `MERGE (n:{entity.type} ...)`.

5. **Generic entity properties are stored as a JSON string in a single Neo4j property field**, rather than as individual node properties. This means generic entities cannot be queried by individual property values using Neo4j indexes.

6. **Exporting while disconnected writes an empty file**, potentially overwriting valid backups.

7. **Inconsistent error contract**: Some methods return `None` on not-found, others raise `ValueError`. No clear pattern.

**What breaks if you change it**:

This is the primary storage adapter. Any Entity or Relationship created in Kosmos flows through this module (in Simple Mode). Cross-module callers: `kosmos/world_model/factory.py:121` (the only import site). Changes to type dispatch logic, Cypher queries, or node conversion code affect all knowledge persistence.

**Runtime Dependencies**:

- `kosmos.knowledge.graph.KnowledgeGraph` (via `get_knowledge_graph()` singleton)
- `kosmos.literature.base_client.PaperMetadata`, `PaperSource`
- `kosmos.world_model.interface.WorldModelStorage`, `EntityManager`
- `kosmos.world_model.models.Entity`, `Relationship`, `Annotation`, `EXPORT_FORMAT_VERSION`

**Public API**:

- **`Neo4jWorldModel.__init__(config)`**: Obtains KnowledgeGraph singleton. All instances share the same database connection.
- **`add_entity(entity) -> str`**: Dispatches to type-specific methods: Paper, Concept, Author, Method, or generic. Soft-fails on disconnect. For Paper entities, creates intermediate `PaperMetadata` with `source=PaperSource.MANUAL` (loses source provenance).
- **`get_entity(entity_id) -> Optional[Entity]`**: Tries `graph.get_paper()` first (4 DB queries), then generic Cypher query. Returns None if not found.
- **`update_entity(entity_id, updates) -> Entity`**: Uses `SET n += $updates` Cypher. Raises `ValueError` if not found (inconsistent with get_entity returning None).
- **`delete_entity(entity_id) -> bool`**: Uses `DETACH DELETE`. Raises `ValueError` if not found.
- **`add_relationship(relationship) -> str`**: Routes CITES and AUTHOR_OF to existing KnowledgeGraph methods; others to generic Cypher.
- **`query_related_entities(entity_id, relationship_type, direction, max_depth) -> List[Entity]`**: Graph traversal. Results limited to 100 (hardcoded).
- **`export_graph(filepath, project) -> Dict`**: Full graph export to JSON. Writes empty file if disconnected.
- **`import_graph(filepath, clear_existing) -> Dict`**: Validates format version. Partial imports can occur silently.
- **`get_statistics(project) -> Dict`**: 5 separate Cypher queries.
- **`reset(project)`**: Calls `graph.clear_graph()` which deletes ALL data. Destructive.
- **`verify_entity(entity_id, verified_by)`**: Sets verified=true on the node.
- **`add_annotation(entity_id, annotation)`**: Appends to JSON array on the node.
- **`get_annotations(entity_id)`**: Deserializes annotations array. Malformed entries skipped.

**Test Coverage**: `tests/e2e/test_world_model.py` (445 lines) tests entity persistence, relationship creation, export/import roundtrip, statistics. `tests/integration/test_world_model_persistence.py` requires Neo4j. Neo4j tests gated behind `@requires_neo4j`.

**Gotchas**:

1. [HIGH] **Parameter mismatch bugs**: `_add_author_entity`, `_add_method_entity`, and `add_relationship` (CITES) pass incorrect parameter names to `KnowledgeGraph` methods. Would raise `TypeError` at runtime.
2. [HIGH] **Cypher injection**: Entity type, relationship type, and project name injected via f-strings without sanitization.
3. [MEDIUM] **Silent data loss on export**: Exporting while disconnected writes an empty file.
4. [MEDIUM] **Inconsistent error contract**: Some methods return `None` on not-found, others raise `ValueError`.
5. [MEDIUM] **Soft-fail on disconnect**: Most methods silently return default values when Neo4j is disconnected.
6. [LOW] **Deprecated `datetime.utcnow()`**: Used in annotation timestamps.

---

### `kosmos/agents/base.py` -- Calibration Module (BaseAgent) (see Change Impact Index: Hub 9)

**Module**: `kosmos/agents/base.py`

**What it does**: Defines `BaseAgent`, the abstract base class for all research agents in Kosmos, plus three Pydantic data models (`AgentMessage`, `AgentState`, `AgentStatus`) and the `MessageType` enum that form the inter-agent communication protocol. Every agent -- `ResearchDirectorAgent`, `HypothesisGeneratorAgent`, `ExperimentDesignerAgent`, `DataAnalystAgent`, `LiteratureAnalyzerAgent` -- inherits from `BaseAgent` (5 subclasses).

**Git Risk**: risk=0.37, churn=4, hotfixes=2, authors=1 -- Low-moderate risk, single author.

**What's non-obvious**:

1. **Dual message queues**: `__init__` creates both a sync `message_queue: List` and an async `_async_message_queue: asyncio.Queue`. `receive_message()` appends to both. The async queue is never consumed by any code in the codebase -- it is populated but never drained.

2. **Dynamic config import inside hot path**: `send_message()` and `receive_message()` each perform a deferred `from kosmos.config import get_config` inside a try/except on every call. This avoids a circular import at module load time but means config availability is silently optional.

3. **`message_handlers` dict is dead code**: `register_message_handler()` populates `self.message_handlers`, but no code in the codebase reads from it. Neither `process_message()` nor any subclass dispatches through this dict.

4. **`_on_pause` / `_on_resume` hooks are never called**: `pause()` and `resume()` change status but never invoke `_on_pause()` or `_on_resume()`. Only `_on_start` and `_on_stop` are actually called.

5. **`execute()` signature contract is violated by 2/5 subclasses**: Base declares `execute(self, task: Dict[str, Any]) -> Dict[str, Any]`. `HypothesisGeneratorAgent.execute()` accepts `AgentMessage` and `ExperimentDesignerAgent.execute()` also takes `AgentMessage`. Polymorphic dispatch through `BaseAgent` will pass wrong type.

6. **`start()` is one-shot**: Guard checks `status != CREATED`. After `stop()`, agent cannot be restarted. Must create a new instance.

7. **`datetime.utcnow()` used throughout**: 7 call sites. Deprecated in Python 3.12+. All timestamps are naive.

**What breaks if you change it**:

- 5 agent subclasses directly inherit `BaseAgent`.
- `AgentRegistry` stores `Dict[str, BaseAgent]`, calls `start()`, `stop()`, `is_running()`, `is_healthy()`, `get_status()`, `set_message_router()`, and reads `message_queue`, `messages_sent`, `messages_received`, `tasks_completed`, `errors_encountered`.
- `AgentMessage` imported by 8 test files and 5 production modules.
- `AgentStatus` enum imported by `registry.py` and 3 test files.
- Renaming any public attribute on `BaseAgent` breaks `AgentRegistry.get_system_health()`.

**Runtime Dependencies**:

- `pydantic`: `BaseModel` and `Field`
- `asyncio`: `asyncio.Queue`, `asyncio.iscoroutine()`, `asyncio.get_running_loop()`, `asyncio.run()`
- `kosmos.config.get_config`: deferred import at runtime
- `uuid`, `logging`

**Public API**:

- **`AgentStatus(str, Enum)`**: 8 values: CREATED, STARTING, RUNNING, IDLE, WORKING, PAUSED, STOPPED, ERROR.
- **`MessageType(str, Enum)`**: 4 values: REQUEST, RESPONSE, NOTIFICATION, ERROR.
- **`AgentMessage(BaseModel)`**: Fields: id (auto UUID), type (MessageType), from_agent, to_agent, content (Dict), correlation_id, timestamp, metadata. `to_dict()` with `timestamp.isoformat()`. `to_json()` raises `TypeError` if content contains non-serializable values.
- **`AgentState(BaseModel)`**: Persistence snapshot with agent_id, agent_type, status, data, created_at, updated_at.
- **`BaseAgent.__init__(agent_id, agent_type, config)`**: All optional. Creates both queues. Status = CREATED.
- **`BaseAgent.start()`**: CREATED -> STARTING -> RUNNING. Calls `_on_start()`. One-shot (no restart). Error in `_on_start()` sets status to ERROR and re-raises.
- **`BaseAgent.stop()`**: Calls `_on_stop()`, sets STOPPED. Can be called in any status.
- **`BaseAgent.pause()`/`resume()`**: Status changes only. Does NOT call hooks.
- **`BaseAgent.is_running()`**: True iff status is exactly RUNNING.
- **`BaseAgent.is_healthy()`**: True if RUNNING, IDLE, or WORKING.
- **`BaseAgent.get_status()`**: Dict snapshot including `message_queue_length`.
- **`async BaseAgent.send_message(to_agent, content, message_type, correlation_id)`**: Creates AgentMessage, increments counter, optionally logs, invokes router. Router failure caught and logged. Config import failure silently passes.
- **`BaseAgent.send_message_sync(...)`**: Sync wrapper with 30s timeout. Detects running loop. Can deadlock from event loop thread.
- **`async BaseAgent.receive_message(message)`**: Appends to both queues, calls `process_message()`. If `process_message()` raises, sends ERROR message back.
- **`async BaseAgent.process_message(message)`**: Default logs warning. Subclasses override.
- **`BaseAgent.register_message_handler(message_type, handler)`**: Dead code -- nothing reads from it.
- **`BaseAgent.set_message_router(router)`**: Sets callback. Called by `AgentRegistry.register()`.
- **`BaseAgent.get_state()`**: Returns `AgentState`. Side effect: mutates `updated_at`.
- **`BaseAgent.restore_state(state)`**: Overwrites 6 attributes. Does not restore queues, counters, or router.
- **`BaseAgent.execute(task)`**: Raises `NotImplementedError`.

**Test Coverage**: No dedicated unit test file. Tested indirectly through `tests/integration/test_async_message_passing.py` (16 test functions covering send, receive, broadcast, routing, concurrent sends). Integration tests require `ANTHROPIC_API_KEY` env var. `AgentMessage` and `MessageType` imported by 6 test files.

**Gotchas**:

1. [HIGH] **`_on_pause`/`_on_resume` hooks never fire**: Despite being defined as override points.
2. [HIGH] **Sync wrappers deadlock if called from the event loop thread**: `send_message_sync()` blocks forever in this scenario.
3. [MEDIUM] **`execute()` signature contract broken by 2 subclasses**: Polymorphic dispatch through `BaseAgent` will pass wrong type.
4. [MEDIUM] **`register_message_handler()` is dead code**: Nothing reads from `message_handlers`.
5. [MEDIUM] **`_async_message_queue` is write-only**: Grows unbounded.
6. [LOW] **`get_state()` has side effect**: Mutates `updated_at`.
7. [LOW] **`start()` is one-shot, no restart**: After `stop()`, must create new instance.
8. [LOW] **`restore_state()` does not restore queues or counters**: Partial restoration.
9. [LOW] **`ResearchDirectorAgent.process_message()` is sync, not async**: Fragile -- relies on CPython implementation detail for `await` on non-coroutine.
## Change Impact Index

This index maps every hub module cluster in Kosmos, ranks blast radius, documents hidden coupling, and identifies the functions whose signature or behavior changes break the most downstream code.

### Blast Radius Rankings

| Rank | Module | Connections | Blast Radius | Riskiest Single Change |
|------|--------|------------|--------------|----------------------|
| 1 | `kosmos.config` (`config.py`) | 54 | CRITICAL | Renaming any config field -- 53 importers access by attribute name [FACT: xray.json imports.graph["kosmos.config"].imported_by has 53 entries] |
| 2 | `kosmos.agents.research_director` | 50 | CRITICAL | Changing `__init__()` or `execute()` signature [FACT: research_director.py:68,2868] |
| 3 | `kosmos.models.hypothesis` | 49 | CRITICAL | Renaming `statement` field -- 48 module blast [FACT: hypothesis.py:52] |
| 4 | `kosmos.core.logging` | 141 | MEDIUM | Changing JSON schema keys [FACT: logging.py:51-82] |
| 5 | `kosmos.core.llm` | 35 | HIGH | Changing `generate()` return type from `str` [FACT: llm.py:361] |
| 6 | `kosmos.models.experiment` | 33 | HIGH | Changing `VariableType` string values [FACT: experiment.py:22-25] |
| 7 | `kosmos.core.workflow` | 28 | HIGH | Removing `WorkflowState` enum values [FACT: workflow.py:19-29] |
| 8 | `kosmos.models.result` | 28 | HIGH | Renaming `supports_hypothesis` [FACT: result.py:174] |
| 9 | `kosmos.agents.base` | 15 | HIGH | Changing `BaseAgent.__init__()` or removing async methods [FACT: base.py:113,246] |
| 10 | `kosmos.orchestration.plan_reviewer` | 5 | LOW | Changing `review_plan()` return type [FACT: plan_reviewer.py:101] |

---

### Cluster 0: Config-Provider-Agent Core (Co-Change Cluster)

Git coupling analysis identifies one dominant cluster [FACT: xray.json git.coupling_clusters cluster_id=0, total_cochanges=52] containing 12 files that change together:

- `kosmos/config.py`
- `kosmos/agents/research_director.py`
- `kosmos/agents/experiment_designer.py`
- `kosmos/agents/hypothesis_generator.py`
- `kosmos/cli/commands/config.py`
- `kosmos/cli/commands/run.py`
- `kosmos/core/providers/anthropic.py`
- `kosmos/core/providers/factory.py`
- `kosmos/core/providers/litellm_provider.py`
- `kosmos/core/providers/openai.py`
- `kosmos/execution/code_generator.py`
- `kosmos/execution/executor.py`

**52 total co-changes** in this cluster [FACT: xray.json git.coupling_clusters cluster_id=0 total_cochanges=52]. Any change to `config.py` historically requires touching 5+ files in this cluster.

---

### Hub 1: kosmos.config (54 connections, CRITICAL) (see Module Index: config.py)

| Hub Module | Imported By (Key Callers) | High-Impact Functions | Signature Change Breaks | Behavior Change Breaks |
|------------|--------------------------|----------------------|------------------------|----------------------|
| `kosmos/config.py` | `core/llm.py`, `core/logging.py`, `core/workflow.py`, `agents/base.py`, `agents/research_director.py`, `db/__init__.py`, all CLI commands, 40+ total files [FACT: config module findings: "Imported by 40 files"] | `get_config(reload=False)` [FACT: config.py:1140-1154] | Renaming any of 16 section fields [FACT: config.py:959-976]; Making optional fields required (ValidationError at startup) [FACT: config module findings]; Removing `reset_config()` [FACT: config.py:1157-1160 breaks test isolation] | Changing `_DEFAULT_CLAUDE_SONNET_MODEL` [FACT: config.py:17 imported by llm.py:20, hypothesis.py:13, experiment.py:15, plan_reviewer.py:27]; Changing `is_cli_mode` [FACT: config.py:80-82 breaks llm.py:179]; Changing `env_file` path [FACT: config.py:979]; Changing `validate_provider_config` [FACT: config.py:1024-1043] |
| | | `reset_config()` [FACT: config.py:1157-1160] | Removing it breaks all test isolation | N/A |
| | | `parse_comma_separated(v)` [FACT: config.py:20-26] | Removing it breaks list fields like `enabled_domains` [FACT: config.py:210] | Changing parse logic breaks env var list handling |
| | | `KosmosConfig.get_active_model()` [FACT: config.py:1045-1054] | Raises ValueError for unknown provider | Changing return value breaks LLM initialization |
| | | `KosmosConfig.normalized_url` [FACT: config.py:269-300] | Removing it breaks DB init | Changing path resolution breaks SQLite location |

**Hidden coupling:**
- `_DEFAULT_CLAUDE_SONNET_MODEL = "claude-sonnet-4-5"` [FACT: config.py:17] is a module-level constant imported directly by 4 modules: `kosmos.core.llm` [FACT: llm.py:20], `kosmos.models.hypothesis` [FACT: hypothesis.py:13], `kosmos.models.experiment` [FACT: experiment.py:15], `kosmos.orchestration.plan_reviewer` [FACT: plan_reviewer.py:27]. Changing this constant silently changes default model names in hypothesis responses and experiment design responses.
- `_DEFAULT_CLAUDE_HAIKU_MODEL = "claude-haiku-4-5"` [FACT: config.py:18] is imported by `llm.py` for auto-model selection [FACT: llm.py:20].
- 53 importers access config fields by attribute chains (e.g., `config.logging.level`, `config.research.enabled_domains`) with no interface abstraction [FACT: config module findings: "53 importers with no interface abstraction"].
- Optional configs (`config.openai`, `config.claude`, `config.anthropic`) can be `None` [FACT: config.py:896-919]. Accessing attributes on `None` raises `AttributeError` -- 53 importers must null-check or trust the provider validator [FACT: config module findings: "config.openai can be None"].
- The `.env` file path is hardcoded relative to config module location [FACT: config.py:979 `env_file=str(Path(__file__).parent.parent / ".env")`].
- `sync_litellm_env_vars` model validator [FACT: config.py:986-1022] is a manual workaround for nested BaseSettings -- removing it breaks all LiteLLM env var loading.
- `get_config()` creates log and ChromaDB directories on first call [FACT: config.py:1067-1076] -- a filesystem side effect on import in some code paths.
- Three separate default domain lists must be kept in sync: `config.py:211` defaults [FACT: config.py:211], `research_director.py:265` defaults [FACT: research_director.py:265], and `cli/commands/config.py:246` valid_domains [FACT: cli/commands/config.py:246]. The codebase already has a discrepancy: `materials` is in `valid_domains` but not in the other two lists.

---

### Hub 2: kosmos.agents.research_director (50 connections, CRITICAL) (see Module Index: research_director.py)

| Hub Module | Imported By (Key Callers) | High-Impact Functions | Signature Change Breaks | Behavior Change Breaks |
|------------|--------------------------|----------------------|------------------------|----------------------|
| `kosmos/agents/research_director.py` | `cli.commands.run`, `kosmos.__init__`, `evaluation.scientific_evaluation`, 15 test modules, 22 total importers [FACT: xray imports.graph research_director imported_by=22] | `__init__(research_question, domain, agent_id, config)` [FACT: research_director.py:68-83] | Changing param names/types breaks CLI, eval, 15 tests | Changing config dict key names (max_iterations, max_runtime_hours, etc.) [FACT: research_director.py:104-112] |
| | | `execute()` [FACT: research_director.py:~2868-2909] | Changing return type breaks all callers | Changing internal workflow routing |
| | | `decide_next_action()` [FACT: research_director.py:~2388-2548] | N/A (internal) | Changes workflow progression for all research sessions |
| | | `_handle_*_action()` methods [FACT: research_director.py:1391-1979] | N/A (internal) | Changes how specific agent tasks are dispatched |
| | | `_send_to_*()` methods [FACT: research_director.py:1039-1219] | N/A (internal) | Changes message-based communication paths |

**Hidden coupling:**
- 7 lazy-loaded agent modules are imported on first use [FACT: research_director.py:145-152]: `hypothesis_generator`, `experiment_designer`, `code_generator`, `executor`, `data_analyst`, `refiner`, `async_llm`/`parallel`. The effective connection count is 50+ when including these.
- Dual communication anti-pattern [FACT: research_director module findings]: message-based path (lines 1039-1219) uses `Dict[str, Any]` content with no schema validation; direct-call path (lines 1391-1979) uses typed method signatures. Both must be maintained simultaneously.
- `agent_registry: Dict[str, str]` [FACT: research_director.py:142] maps agent types to IDs. Lazy init means unregistered targets fail silently.
- Asyncio locks [FACT: research_director.py:192-195] AND threading locks [FACT: research_director.py:198-200] coexist: `_research_plan_lock`, `_strategy_stats_lock`, `_workflow_lock`, `_agent_registry_lock` (async) plus `_research_plan_lock_sync`, `_strategy_stats_lock_sync`, `_workflow_lock_sync` (threading).
- Error recovery constants are hard-coded: `MAX_CONSECUTIVE_ERRORS = 3` [FACT: research_director.py:45], `ERROR_BACKOFF_SECONDS = [2, 4, 8]` [FACT: research_director.py:46], `MAX_ACTIONS_PER_ITERATION = 50` [FACT: research_director.py:50].
- World model init is wrapped in try/except [FACT: research_director.py:242-255], setting `self.wm = None` on failure -- the director operates without world model gracefully but with degraded knowledge persistence.
- `_send_to_convergence_detector()` is DEPRECATED [FACT: research_director module findings: "file:1986-1988"] -- dead code that will confuse maintainers.
- `_on_start()` hook [FACT: research_director.py:319-332] transitions workflow to `GENERATING_HYPOTHESES` as a side effect -- callers of `start()` may not expect a state transition.
- At 2953 lines [FACT: research_director module findings: "At 2953 lines"], this is the largest agent file.

---

### Hub 3: kosmos.models.hypothesis (49 connections, CRITICAL) (see Module Index: hypothesis.py)

| Hub Module | Imported By (Key Callers) | High-Impact Functions | Signature Change Breaks | Behavior Change Breaks |
|------------|--------------------------|----------------------|------------------------|----------------------|
| `kosmos/models/hypothesis.py` | All agents, convergence, feedback, memory, code_generator, 14 experiment templates, world model -- 48 importers [FACT: xray imports.graph["kosmos.models.hypothesis"].imported_by has 48 entries] | `Hypothesis.__init__()` [FACT: hypothesis.py:32-156] | Renaming `statement` (48 importers), renaming score fields (breaks convergence, prioritization) [FACT: hypothesis.py:59-62], making `id` required (breaks constructors) [FACT: hypothesis.py:50] | Tightening validators rejects LLM output that passed before [FACT: hypothesis.py:86-103]; Changing `ConfigDict(use_enum_values=False)` to `True` changes all serialization [FACT: hypothesis.py:156] |
| | | `Hypothesis.to_dict()` [FACT: hypothesis.py:117-142] | Changing return keys breaks DB serialization | Changing enum-to-value conversion |
| | | `ExperimentType` enum [FACT: hypothesis.py:15-19] | Removing values breaks template matching in 14 modules | Adding values is safe if existing preserved |
| | | `HypothesisStatus` enum [FACT: hypothesis.py:22-29] | Removing values breaks convergence, research_director, 15 tests | N/A |
| | | `is_testable(threshold=0.3)` [FACT: hypothesis.py:144-148] | N/A | Changing default threshold changes which hypotheses proceed |
| | | `is_novel(threshold=0.5)` [FACT: hypothesis.py:150-154] | N/A | Changing default threshold changes novelty filtering |

**Hidden coupling:**
- `to_dict()` and `.model_dump()` produce different output [FACT: hypothesis module findings: "use_enum_values=False"]. `to_dict()` manually converts enums to `.value` strings; `.model_dump()` returns enum objects. Code using `model_dump()` without `mode="json"` gets wrong types.
- `None` scores fail threshold checks silently [FACT: hypothesis module findings: "None scores fail threshold checks"]. Newly created hypotheses with no scores fail both `is_testable()` and `is_novel()` even if they are actually testable and novel.
- Inconsistent novelty thresholds across three locations: `NoveltyReport.novelty_threshold_used=0.75` [FACT: hypothesis.py:259], `HypothesisGenerationRequest.min_novelty_score=0.5` [FACT: hypothesis.py:184], `Hypothesis.is_novel(threshold=0.5)` [FACT: hypothesis.py:150].
- `_DEFAULT_CLAUDE_SONNET_MODEL` imported from config [FACT: hypothesis.py:13] couples the hypothesis model to config module's model name constant.
- `datetime.utcnow` usage [FACT: hypothesis.py:76-77] is deprecated in Python 3.12+.
- `HypothesisGenerationResponse.get_best_hypothesis()` silently returns the first hypothesis as fallback [FACT: hypothesis.py:229] when none have `priority_score`.
- Dual model system: Pydantic runtime `Hypothesis` + SQLAlchemy persistence `HypothesisModel` in `kosmos.db.models` with no automatic conversion [FACT: hypothesis module findings: "Dual model system"].

---

### Hub 4: kosmos.core.logging (141 connections, MEDIUM) (see Module Index: core_logging.py)

| Hub Module | Imported By (Key Callers) | High-Impact Functions | Signature Change Breaks | Behavior Change Breaks |
|------------|--------------------------|----------------------|------------------------|----------------------|
| `kosmos/core/logging.py` | 140 modules [FACT: xray imports.graph kosmos.core.logging imported_by=140] | `setup_logging(level, log_format, log_file, debug_mode)` [FACT: logging.py:133-217] | Renaming params breaks `configure_from_config()` [FACT: logging.py:135-137] | Removing handler clearing [FACT: logging.py:179] causes duplicate log lines |
| | | `get_logger(name)` [FACT: logging.py:220-239] | N/A (thin wrapper) | Changing return type from `logging.Logger` |
| | | `configure_from_config()` [FACT: logging.py:397-403] | N/A (internal) | Changing config field reads |
| | | `ExperimentLogger.get_summary()` [FACT: logging.py:364-380] | Changing return dict keys | N/A |

**Hidden coupling:**
- True coupling is through stdlib root logger [FACT: logging.py:239 returns `logging.getLogger(name)`]. The 141 import connections overstate direct dependency -- most modules use stdlib logging directly.
- JSON output schema at [FACT: logging.py:51-82] is a de facto public API. Keys: `timestamp` [FACT: logging.py:52], `level` [FACT: logging.py:53], `logger` [FACT: logging.py:54], `message` [FACT: logging.py:55], `module` [FACT: logging.py:56], `function` [FACT: logging.py:57], `line` [FACT: logging.py:58]. Optional: `workflow_id` [FACT: logging.py:72], `cycle` [FACT: logging.py:74], `task_id` [FACT: logging.py:76].
- `correlation_id` ContextVar [FACT: logging.py:23-25] is read at [FACT: logging.py:62-64] but never set in-module -- set externally.
- `debug_mode` silently overrides `level` [FACT: logging.py:174-175].
- `LogFormat` enum has only two values: `JSON = "json"` [FACT: logging.py:30] and `TEXT = "text"` [FACT: logging.py:31] -- adding a third requires changing the setup function.
- `RotatingFileHandler` uses 10MB max [FACT: logging.py:201] and 5 backups [FACT: logging.py:203].
- `configure_from_config()` lazy-imports `get_config` [FACT: logging.py:395] creating a circular dependency risk.

---

### Hub 5: kosmos.core.llm (35 connections, HIGH) (see Module Index: core_llm.py)

| Hub Module | Imported By (Key Callers) | High-Impact Functions | Signature Change Breaks | Behavior Change Breaks |
|------------|--------------------------|----------------------|------------------------|----------------------|
| `kosmos/core/llm.py` | 27 importers: all agents, code_generator, analysis modules [FACT: xray imports.graph llm imported_by=27] | `get_client(reset, use_provider_system)` [FACT: llm.py:613] | Renaming breaks all agent `__init__` methods | Changing fallback provider |
| | | `ClaudeClient.generate(prompt, system, max_tokens, temperature, ...)` [FACT: llm.py:207-216] | Changing return type from `str` breaks 27 callers [FACT: llm.py:361] | Changing auto-model selection thresholds [FACT: llm.py:251-268]; Removing cache [FACT: llm.py:277-300] |
| | | `ClaudeClient.generate_structured()` [FACT: llm.py:410] | Changing return type from `Dict` [FACT: llm.py:471] | Changing retry behavior [FACT: llm.py:460-476] |
| | | `ClaudeClient.generate_with_messages()` [FACT: llm.py:367] | Changing return type from `str` [FACT: llm.py:404] | N/A |
| | | `get_usage_stats()` [FACT: llm.py:488] | Changing return dict keys [FACT: llm.py:529-538] | N/A |

**Hidden coupling:**
- `kosmos.execution.code_generator` directly imports `ClaudeClient` class [FACT: code_generator.py:19 `from kosmos.core.llm import ClaudeClient`] -- hard dependency on class name, not on the `get_client()` abstraction.
- Thread-safe via `_client_lock` [FACT: llm.py:610] with double-checked locking [FACT: llm.py:643-649], but thread safety covers only initialization -- subsequent usage of the client instance is not synchronized.
- Two `ClaudeClient` classes exist: `llm.py:108` and `providers/anthropic.py` alias [FACT: llm module findings]. Import path determines which class is used.
- `ModelComplexity` has 20 `COMPLEX_KEYWORDS` [FACT: llm.py:46-50] biased toward research prompts -- this silently routes all research prompts to Sonnet.
- Cache key includes `system`, `max_tokens`, `temperature`, `stop_sequences` [FACT: llm.py:278-283] -- same prompt with different params misses cache.
- `generate_with_messages()` ignores auto model selection [FACT: llm module findings] -- always uses `self.model`, not complexity-based selection.
- Cost estimation hardcodes "claude-sonnet-4-5" pricing [FACT: llm.py:519] even when other models are used.

---

### Hub 6: kosmos.models.experiment (33 connections, HIGH) (see Module Index: experiment.py)

| Hub Module | Imported By (Key Callers) | High-Impact Functions | Signature Change Breaks | Behavior Change Breaks |
|------------|--------------------------|----------------------|------------------------|----------------------|
| `kosmos/models/experiment.py` | 30 importers: experiment_designer, code_generator, result_collector, 14 templates [FACT: xray imports.graph["kosmos.models.experiment"].imported_by has 30 entries] | `ExperimentProtocol.__init__()` [FACT: experiment.py:329-575] | Renaming `steps` breaks code_generator [FACT: experiment.py:363]; Renaming `variables` breaks code_generator's `.values()` access [FACT: experiment.py:364] | Changing step validator to reject non-sequential numbering more strictly [FACT: experiment.py:424-438] |
| | | `ExperimentProtocol.to_dict()` [FACT: experiment.py:471-573] | Changing return structure (100-line manual serialization) | N/A |
| | | `VariableType` enum [FACT: experiment.py:22-25] | Changing string values breaks code_generator.py:74 string comparison | N/A |
| | | `StatisticalTest` enum [FACT: experiment.py:31-39] | Removing values | N/A |
| | | `ProtocolStep` model [FACT: experiment.py:138-191] | Removing `code_template` or `library_imports` breaks code_generator | `ensure_title` replacing bad titles silently [FACT: experiment.py:181] |

**Hidden coupling:**
- Code generator's string-based enum comparisons are particularly brittle [FACT: code_generator.py:65-66 `'t_test' in test_type_str.lower()`] and [FACT: code_generator.py:74 filters by `v.type.value == 'independent'`].
- 6-level nesting (ExperimentProtocol -> ProtocolStep -> code_template, ExperimentProtocol -> Variable -> VariableType, etc.) means changes at any level ripple.
- `_MAX_SAMPLE_SIZE = 100_000` [FACT: experiment.py:19] is enforced by validators on both `ControlGroup.sample_size` and `ExperimentProtocol.sample_size`.
- Duplicated validators: `ControlGroup.coerce_sample_size` [FACT: experiment.py:106-125] and `ExperimentProtocol.coerce_protocol_sample_size` [FACT: experiment.py:373-392] are near-identical.
- `ExperimentType` is imported from `kosmos.models.hypothesis` [FACT: experiment.py:14] -- cross-module enum creates a dependency bridge between hypothesis and experiment models.
- `use_enum_values=False` [FACT: experiment.py:575] means `model_dump()` keeps enum objects, not strings.

---

### Hub 7: kosmos.core.workflow (28 connections, HIGH) (see Module Index: core_workflow.py)

| Hub Module | Imported By (Key Callers) | High-Impact Functions | Signature Change Breaks | Behavior Change Breaks |
|------------|--------------------------|----------------------|------------------------|----------------------|
| `kosmos/core/workflow.py` | 26 importers: research_director (primary), tests [FACT: xray imports.graph workflow imported_by=26] | `ResearchWorkflow.transition_to(target_state, action, metadata)` [FACT: workflow.py:260-328] | Changing raises ValueError for invalid transitions -- removing that breaks error handling [FACT: workflow.py:280-284] | Changing `ALLOWED_TRANSITIONS` [FACT: workflow.py:175-227] causes ValueError in previously valid paths |
| | | `ResearchPlan` mutation methods [FACT: workflow.py:100-145] | Renaming fields [FACT: workflow.py:63-94] | Removing deduplication changes idempotency [FACT: workflow.py:102]; Changing `mark_supported()` to not call `mark_tested()` [FACT: workflow.py:116] |
| | | `WorkflowState` enum [FACT: workflow.py:18-29] | Removing values breaks research_director, convergence, 15 tests | N/A |
| | | `ResearchWorkflow.to_dict()` [FACT: workflow.py:351-365] | Changing return keys | N/A |
| | | `get_state_statistics()` [FACT: workflow.py:398-416] | Changing return dict keys [FACT: workflow.py:411-414] | N/A |

**Hidden coupling:**
- `mark_supported()` and `mark_rejected()` both implicitly call `mark_tested()` [FACT: workflow.py:116]. Callers depend on this cascading behavior.
- `transition_to()` lazy-imports `get_config()` [FACT: workflow.py:295-296] and reads `config.logging.log_workflow_transitions` [FACT: workflow.py:296] -- this is the only config dependency and it is failure-tolerant.
- `WorkflowTransition` uses `ConfigDict(use_enum_values=True)` [FACT: workflow.py:48] -- enums serialize as strings in transition records, opposite of the hypothesis model which uses `use_enum_values=False`.
- CONVERGED is nearly terminal -- only one exit transition exists (back to GENERATING_HYPOTHESES) [FACT: workflow.py:211-213]. There is no CONVERGED->PAUSED or CONVERGED->ERROR path.
- No persistence mechanism -- workflow state exists only in memory [FACT: workflow module findings]. Process crashes lose all transition history.
- Duration calculations are O(n^2) in the worst case [FACT: workflow module findings: "get_state_duration does a nested scan"]. Long-running sessions with many transitions become slow.
- `transition_history` grows unboundedly [FACT: workflow module findings: "No maximum history size"].

---

### Hub 8: kosmos.models.result (28 connections, HIGH) (see Module Index: result.py)

| Hub Module | Imported By (Key Callers) | High-Impact Functions | Signature Change Breaks | Behavior Change Breaks |
|------------|--------------------------|----------------------|------------------------|----------------------|
| `kosmos/models/result.py` | 27 importers: research_director, data_analyst, summarizer, convergence, feedback, result_collector, visualization, memory [FACT: xray imports.graph["kosmos.models.result"].imported_by has 27 entries] | `ExperimentResult.__init__()` [FACT: result.py:127-278] | Renaming `supports_hypothesis` breaks convergence and research_director [FACT: result.py:174]; Renaming `primary_p_value` breaks significance testing [FACT: result.py:170]; Making `metadata` optional breaks all `metadata.X` access [FACT: result.py:180] | Changing `to_dict()` to `exclude_none=False` changes output shape [FACT: result.py:243] |
| | | `to_dict()` / `from_dict()` round-trip [FACT: result.py:241-257] | Removing `from_dict()` or `from_json()` breaks deserialization of persisted data | N/A |
| | | `is_significant(alpha=0.05)` [FACT: result.py:259-263] | N/A | Changing default alpha |
| | | `ResultStatus` enum [FACT: result.py:17-22] | Changing values breaks feedback patterns | N/A |
| | | `get_summary_stats()` [FACT: result.py:265-277] | Changing return dict keys [FACT: result.py:270-276] | N/A |

**Hidden coupling:**
- `to_dict()`/`from_dict()` round-trip is a database serialization boundary [FACT: model_layer findings]. Field renames require schema migration for persisted data.
- `supports_hypothesis` is tri-state: `True`, `False`, or `None` [FACT: result.py:174-177]. Convergence detector counts `None` differently from `False`.
- `to_dict()` delegates to `model_to_dict(self, mode='json', exclude_none=True)` from `kosmos.utils.compat` [FACT: result.py:241-243] -- unlike hypothesis and experiment models which have manual `to_dict()`. Different serialization strategy for each model.
- `to_json()` uses Pydantic v2-native `model_dump_json()` [FACT: result.py:245-247] which may produce slightly different output than `to_dict()` for edge cases (datetime formatting).
- `export_markdown()` will crash on `None` stats [FACT: result module findings: ":.2f format strings do not handle None values"]. Latent bug for incomplete variable results.
- `StatisticalTestResult` requires all three significance fields (`significant_0_05`, `significant_0_01`, `significant_0_001`) with no defaults [FACT: result.py:83-85]. Any code constructing one must compute all three levels.
- Dual serialization: `ExperimentResult.to_dict()` uses compat layer; `Hypothesis.to_dict()` is manual; `ExperimentProtocol.to_dict()` is 100-line manual. Three different strategies across three pipeline models.

---

### Hub 9: kosmos.agents.base (15 connections, HIGH) (see Module Index: kosmos/agents/base.py)

| Hub Module | Imported By (Key Callers) | High-Impact Functions | Signature Change Breaks | Behavior Change Breaks |
|------------|--------------------------|----------------------|------------------------|----------------------|
| `kosmos/agents/base.py` | 15 importers: 5 agent subclasses, registry, 8 test modules [FACT: xray imports.graph base imported_by=15] | `BaseAgent.__init__(agent_id, agent_type, config)` [FACT: base.py:113-153] | Changing params breaks all `super().__init__()` calls in 5 subclasses | N/A |
| | | `execute(task: Dict) -> Dict` [FACT: base.py:485-497] | Removing it breaks 4 subclasses | N/A |
| | | `send_message()` / `send_message_sync()` [FACT: base.py:246-327] | Removing breaks research_director coordination | N/A |
| | | `get_status() -> Dict` [FACT: base.py:219-240] | Changing return dict keys breaks 8 test modules | N/A |
| | | `AgentMessage` model [FACT: base.py:45-84] | Changing `content` type from `Dict[str, Any]` | N/A |

**Hidden coupling:**
- Agent interface mismatch: Base defines `execute(task: Dict) -> Dict` [FACT: base.py:485-497] but subclasses override as `execute(message: AgentMessage) -> AgentMessage` [FACT: hypothesis_generator.py:91, experiment_designer.py:109]. No type enforcement.
- `AgentStatus` enum has 8 values [FACT: base.py:26-34]. `is_healthy()` returns True only for RUNNING, IDLE, and WORKING [FACT: base.py:211-217] -- forgetting to set status back to IDLE causes health check failures.
- `_async_message_queue: asyncio.Queue` [FACT: base.py:137] and `message_queue: List[AgentMessage]` [FACT: base.py:136] coexist -- legacy sync plus new async.
- `set_message_router()` [FACT: base.py:417-433] is called by `AgentRegistry.register()` [FACT: registry.py:94] to wire up message routing. The router callback signature is a hidden contract.

---

### Hub 10: kosmos.orchestration.plan_reviewer (5 connections, LOW) (see Module Index: plan_reviewer.py)

| Hub Module | Imported By (Key Callers) | High-Impact Functions | Signature Change Breaks | Behavior Change Breaks |
|------------|--------------------------|----------------------|------------------------|----------------------|
| `kosmos/orchestration/plan_reviewer.py` | 5 importers: `research_loop.py`, 4 test modules [FACT: xray imports.graph plan_reviewer imported_by=5] | `review_plan(plan: Dict, context: Dict) -> PlanReview` [FACT: plan_reviewer.py:101-162] | Changing signature breaks research_loop and tests | N/A |
| | | `PlanReview.approved` [FACT: plan_reviewer.py:34] | Removing it breaks execution gating | Changing approval logic |
| | | Mock mode [FACT: plan_reviewer.py:316-357] | Removing mock breaks testing without API | Changing mock base scores [FACT: plan_reviewer.py:327-328] |

**Hidden coupling:**
- NOT a BaseAgent subclass [FACT: plan_reviewer.py:55] -- standalone utility with no lifecycle, messaging, or async support. If someone inherits `BaseAgent`, it changes the entire interface.
- 5 review dimensions scored 0-10 with weights [FACT: plan_reviewer.py:69-75]: `specificity` (0.25), `relevance` (0.25), `novelty` (0.20), `coverage` (0.15), `feasibility` (0.15). Weights are defined but not currently used in scoring [FACT: plan_reviewer module findings: "DIMENSION_WEIGHTS not currently used"].
- Structural requirements are hardcoded: >= 3 `data_analysis` tasks [FACT: plan_reviewer.py:288-294], >= 2 task types [FACT: plan_reviewer.py:297-302]. Changing these silently rejects previously-approved plans.
- `anthropic_client=None` triggers deterministic mock mode [FACT: plan_reviewer.py:117-118]. Mock returns base score 7.5 if structural pass, 6.0 if not [FACT: plan_reviewer.py:327-328].

---

### Cross-Hub Coupling Map

```
kosmos.config ----[53 importers]----> ALL MODULES
     |
     +-- _DEFAULT_CLAUDE_SONNET_MODEL --> llm.py, hypothesis.py, experiment.py, plan_reviewer.py
     +-- config.logging --> logging.py, workflow.py, base.py
     +-- config.research --> research_director.py
     +-- config.database --> db/__init__.py

kosmos.models.hypothesis --[ExperimentType enum]--> kosmos.models.experiment
     |                                                     |
     +-- Hypothesis.id ----[FK: hypothesis_id]--------> ExperimentProtocol.hypothesis_id
                                                              |
                                                              +--[FK: protocol_id]---> ExperimentResult.protocol_id
     |
     +-- Hypothesis.id ----[FK: hypothesis_id]--------> ExperimentResult.hypothesis_id

research_director --[lazy import]--> hypothesis_generator, experiment_designer,
                                     code_generator, executor, data_analyst, refiner
```
## Key Interfaces

### Agent Framework

#### BaseAgent (Abstract Base Class)
- `BaseAgent.__init__(agent_id=None, agent_type=None, config=None)` -- Initializes agent with optional UUID, type name (defaults to class name), and config dict; creates both sync and async message queues; sets status to CREATED [FACT] (kosmos/agents/base.py:113-153)
- `BaseAgent.start()` -- Transitions status CREATED -> STARTING -> RUNNING and calls `_on_start()` hook; no-op if status is not CREATED, making it one-shot with no restart path [FACT] (kosmos/agents/base.py:159-175)
- `BaseAgent.stop()` -- Calls `_on_stop()` hook then sets status to STOPPED; callable from any status [FACT] (kosmos/agents/base.py:177-188)
- `BaseAgent.pause()` -- Sets status to PAUSED if currently RUNNING; does NOT call `_on_pause()` despite defining it [FACT] (kosmos/agents/base.py:189-196)
- `BaseAgent.resume()` -- Sets status to RUNNING if currently PAUSED; does NOT call `_on_resume()` despite defining it [FACT] (kosmos/agents/base.py:198-205)
- `BaseAgent.is_running() -> bool` -- Returns True only if status is exactly RUNNING; WORKING and IDLE return False [FACT] (kosmos/agents/base.py:207-209)
- `BaseAgent.is_healthy() -> bool` -- Returns True if status is RUNNING, IDLE, or WORKING; designed for subclass override [FACT] (kosmos/agents/base.py:211-217)
- `BaseAgent.get_status() -> dict` -- Returns dict snapshot of agent state including statistics and message_queue_length from sync queue [FACT] (kosmos/agents/base.py:219-240)
- `async BaseAgent.send_message(to_agent, content, message_type, correlation_id) -> AgentMessage` -- Creates AgentMessage, increments messages_sent, invokes _message_router if set; router failure is caught and logged, never raises [FACT] (kosmos/agents/base.py:246-300)
- `BaseAgent.send_message_sync(...)` -- Sync wrapper; detects running loop and uses run_coroutine_threadsafe with 30s timeout, or falls back to asyncio.run(); will deadlock if called from the event loop thread [FACT] (kosmos/agents/base.py:302-327)
- `async BaseAgent.receive_message(message: AgentMessage)` -- Appends to both sync and async queues, calls process_message(); on failure sends ERROR message back to sender using correlation_id [FACT] (kosmos/agents/base.py:329-367)
- `async BaseAgent.process_message(message: AgentMessage)` -- Default logs warning that processing is not implemented; subclasses must override [FACT] (kosmos/agents/base.py:382-391)
- `BaseAgent.register_message_handler(message_type, handler)` -- Stores handler in self.message_handlers; this dict is never read by any code in the codebase (dead code) [FACT] (kosmos/agents/base.py:406-415)
- `BaseAgent.set_message_router(router: Callable)` -- Sets _message_router callback accepting sync or async callable; called by AgentRegistry.register() [FACT] (kosmos/agents/base.py:417-433)
- `BaseAgent.get_state() -> AgentState` -- Creates AgentState persistence snapshot; side effect: mutates self.updated_at [FACT] (kosmos/agents/base.py:439-454)
- `BaseAgent.restore_state(state: AgentState)` -- Overwrites 6 instance attributes from saved state; does not restore queues, counters, or message router [FACT] (kosmos/agents/base.py:456-470)
- `BaseAgent.save_state_data(key, value)` -- Stores key-value in state_data dict and updates updated_at [FACT] (kosmos/agents/base.py:472-475)
- `BaseAgent.get_state_data(key, default=None)` -- Dict .get() on state_data; pure read [FACT] (kosmos/agents/base.py:477-479)
- `BaseAgent.execute(task: Dict) -> Dict` -- Raises NotImplementedError; subclasses must override [FACT] (kosmos/agents/base.py:485-497)

#### Agent Communication Models
- `AgentMessage(BaseModel)` -- Pydantic model with fields: id (auto UUID), type (MessageType), from_agent, to_agent, content (Dict), correlation_id, timestamp, metadata [FACT] (kosmos/agents/base.py:45-68)
- `AgentMessage.to_dict() -> dict` -- Returns dict with timestamp as isoformat string [FACT] (kosmos/agents/base.py:69-80)
- `AgentMessage.to_json() -> str` -- Returns json.dumps of to_dict(); raises TypeError if content contains non-serializable values [FACT] (kosmos/agents/base.py:82-84)
- `AgentState(BaseModel)` -- Pydantic model for persistence snapshots with agent_id, agent_type, status, data, created_at, updated_at [FACT] (kosmos/agents/base.py:87-94)
- `MessageType(str, Enum)` -- 4 values: REQUEST, RESPONSE, NOTIFICATION, ERROR [FACT] (kosmos/agents/base.py:37-42)
- `AgentStatus(str, Enum)` -- 8 values: CREATED, STARTING, RUNNING, IDLE, WORKING, PAUSED, STOPPED, ERROR [FACT] (kosmos/agents/base.py:25-34)

#### AgentRegistry (Message Router)
- `AgentRegistry.register(agent: BaseAgent)` -- Stores agent in _agents dict and calls agent.set_message_router(self._route_message) to wire up automatic delivery [FACT] (kosmos/agents/registry.py:57-97)
- `AgentRegistry._route_message(message: AgentMessage)` -- Looks up message.to_agent and calls target.receive_message(); silently drops message with warning if target not found [FACT] (kosmos/agents/registry.py:230-247)

### Research Director (Orchestrator)

- `ResearchDirectorAgent.__init__(research_question, domain, agent_id=None, config=None)` -- Initializes director with research question, creates workflow state machine, convergence detector, LLM client, optional world model; attempts DB init in constructor [FACT] (kosmos/agents/research_director.py:68-260)
- `async ResearchDirectorAgent.execute(task: Dict) -> dict` -- Entry point supporting "start_research" (generates plan, starts workflow) and "step" (single research step) task types [FACT] (kosmos/agents/research_director.py:2868-2909)
- `ResearchDirectorAgent.execute_sync(task: Dict) -> dict` -- Synchronous wrapper; tries running event loop, falls back to asyncio.run() [FACT] (kosmos/agents/research_director.py:2911-2920)
- `ResearchDirectorAgent.decide_next_action() -> NextAction` -- Core decision engine checking budget, runtime limits, loop guard (50 actions max), then state-based decision tree; defaults to GENERATE_HYPOTHESIS for unknown states [FACT] (kosmos/agents/research_director.py:2388-2548)
- `ResearchDirectorAgent.process_message(message: AgentMessage)` -- Routes incoming messages by sender agent_type string to type-specific handlers; logs warning for unknown types; is sync despite base declaring async [FACT] (kosmos/agents/research_director.py:568-593)
- `ResearchDirectorAgent.generate_research_plan() -> str` -- Uses LLM to generate initial research plan, stores in research_plan.initial_strategy [FACT] (kosmos/agents/research_director.py:2349-2382)
- `ResearchDirectorAgent.get_research_status() -> dict` -- Returns comprehensive status with workflow state, iteration counts, hypothesis/experiment/result counts, strategy stats [FACT] (kosmos/agents/research_director.py:2926-2952)
- `ResearchDirectorAgent.execute_experiments_batch(protocol_ids) -> List[dict]` -- Parallel experiment execution when concurrent mode enabled; sequential fallback uses deprecated asyncio.get_event_loop().run_until_complete() [FACT] (kosmos/agents/research_director.py:2152-2204)
- `async ResearchDirectorAgent.evaluate_hypotheses_concurrently(hypothesis_ids) -> List[dict]` -- Concurrent hypothesis evaluation via AsyncClaudeClient.batch_generate(); returns empty list if async client unavailable [FACT] (kosmos/agents/research_director.py:2206-2275)
- `async ResearchDirectorAgent.analyze_results_concurrently(result_ids) -> List[dict]` -- Concurrent result analysis via AsyncClaudeClient.batch_generate() [FACT] (kosmos/agents/research_director.py:2277-2343)
- `ResearchDirectorAgent._handle_error_with_recovery(error_source, error_message, recoverable, error_details) -> Optional[NextAction]` -- Exponential backoff (2, 4, 8 seconds) with circuit breaker after 3 consecutive errors; uses blocking time.sleep() in async context [FACT] (kosmos/agents/research_director.py:599-684)

### Orchestration

#### DelegationManager (Task Routing)
- `DelegationManager.AGENT_ROUTING` -- Maps task types to agent names: data_analysis->DataAnalystAgent, literature_review->LiteratureAnalyzerAgent, hypothesis_generation->HypothesisGeneratorAgent, experiment_design->ExperimentDesignerAgent [FACT] (kosmos/orchestration/delegation.py:89-95)
- `DelegationManager.execute_plan(plan) -> results` -- Splits tasks into batches (max 3 parallel), executes each via asyncio.gather(), classifies results as completed/failed [FACT] (kosmos/orchestration/delegation.py:126-194)

#### PlanReviewerAgent
- `PlanReviewerAgent.__init__(anthropic_client, model=None, min_average_score=7.0, min_dimension_score=5.0, temperature=0.3)` -- Accepts optional client; None triggers mock mode; model defaults to claude-sonnet-4-5 [FACT] (kosmos/orchestration/plan_reviewer.py:77)
- `PlanReviewerAgent.review_plan(plan: Dict, context: Dict) -> PlanReview` -- Primary entry point; dual-path: if client is None delegates to _mock_review(), otherwise calls LLM; approval requires avg score >= 7.0 AND min score >= 5.0 AND structural requirements; on LLM failure falls back to _mock_review() silently [FACT] (kosmos/orchestration/plan_reviewer.py:101-162)
- `PlanReviewerAgent.get_approval_statistics(reviews: List[PlanReview]) -> Dict` -- Computes batch statistics over a list of reviews: approval rate, per-dimension averages, overall average; pure computation [FACT] (kosmos/orchestration/plan_reviewer.py:359-389)

#### PlanReview (dataclass)
- `PlanReview` -- Container with fields: approved (bool), scores (Dict[str, float]), average_score, min_score, feedback, required_changes, suggestions [FACT] (kosmos/orchestration/plan_reviewer.py:32-52)
- `PlanReview.to_dict() -> dict` -- Serializes all fields; no from_dict() counterpart [FACT] (kosmos/orchestration/plan_reviewer.py:32-52)

### Event System

#### EventBus (Pub/Sub)
- `EventBus.subscribe(event_type, callback)` -- Registers sync or async callback for event type; thread-safe with locks [FACT] (kosmos/core/event_bus.py:28-56)
- `EventBus.publish_sync(event)` -- Publishes event to all matching subscribers synchronously [FACT] (kosmos/core/event_bus.py:28-56)
- `EventType(Enum)` -- 16 event types across 5 categories: workflow lifecycle (4), research cycle (3), task (4), LLM calls (4), code execution (5), stage tracking (3) [FACT] (kosmos/core/events.py:15-52)

#### StageTracker
- `StageTracker._publish_to_event_bus(event)` -- Converts internal stage events to StreamingEvent objects and publishes to get_event_bus().publish_sync(); failures caught and logged at debug level [FACT] (kosmos/core/stage_tracker.py:183-220)

### LLM Providers

#### LLMProvider (Abstract Base Class)
- `LLMProvider.__init__(config: Dict)` -- Derives provider_name from class name (strip "Provider", lowercase); initializes 4 non-thread-safe usage counters [FACT] (kosmos/core/providers/base.py:179-195)
- `LLMProvider.generate(prompt, system=None, max_tokens=4096, temperature=0.7, stop_sequences=None, **kwargs) -> LLMResponse` -- Abstract; synchronous LLM generation [FACT] (kosmos/core/providers/base.py:197-224)
- `LLMProvider.generate_async(prompt, system, max_tokens, temperature, stop_sequences, **kwargs) -> LLMResponse` -- Abstract; async generation [FACT] (kosmos/core/providers/base.py:226-253)
- `LLMProvider.generate_with_messages(messages: List[Message], max_tokens=4096, temperature=0.7, **kwargs) -> LLMResponse` -- Abstract; multi-turn conversations [FACT] (kosmos/core/providers/base.py:255-278)
- `LLMProvider.generate_structured(prompt, schema: Dict, system=None, max_tokens=4096, temperature=0.7, **kwargs) -> Dict` -- Abstract; JSON output; returns Dict not LLMResponse, so callers cannot access token usage from the return value [FACT] (kosmos/core/providers/base.py:280-308)
- `LLMProvider.generate_stream(prompt, system, max_tokens, temperature, **kwargs) -> Iterator` -- Raises NotImplementedError by default; overridden by AnthropicProvider and LiteLLMProvider [FACT] (kosmos/core/providers/base.py:310-334)
- `LLMProvider.get_usage_stats() -> dict` -- Returns cumulative stats dict with total_input_tokens, total_output_tokens, total_cost_usd, request_count [FACT] (kosmos/core/providers/base.py:375-389)
- `LLMProvider._update_usage_stats(usage: UsageStats)` -- Increments counters after each API call; protected method; not thread-safe [FACT] (kosmos/core/providers/base.py:391-402)
- `LLMProvider.reset_usage_stats()` -- Zeroes all counters [FACT] (kosmos/core/providers/base.py:404-410)
- `LLMProvider.get_model_info() -> dict` -- Abstract; returns dict with name, max_tokens, cost_per_input_token, cost_per_output_token [FACT] (kosmos/core/providers/base.py:365-373)

#### Provider Data Models
- `Message(role, content, name=None, metadata=None)` -- Conversation message dataclass; role is a bare str, not an enum [FACT] (kosmos/core/providers/base.py:17-31)
- `UsageStats(input_tokens, output_tokens, total_tokens, cost_usd=None, model=None, provider=None, timestamp=None)` -- Token accounting; total_tokens is NOT auto-computed from input+output [FACT] (kosmos/core/providers/base.py:34-54)
- `LLMResponse(content, usage, model, finish_reason=None, raw_response=None, metadata=None)` -- Core return type with 25 string-delegation methods (strip, split, find, etc.) that return str not LLMResponse, losing metadata; isinstance(response, str) returns False [FACT] (kosmos/core/providers/base.py:57-154)
- `ProviderAPIError(provider, message, status_code=None, raw_error=None, recoverable=True)` -- Custom exception; is_recoverable() classifies errors via status code and message pattern matching; recoverable patterns checked first, creating retry-favoring bias [FACT] (kosmos/core/providers/base.py:417-484)

#### Provider Factory
- `get_provider_from_config(config) -> LLMProvider` -- Bridges KosmosConfig to providers; handles backward compatibility where "claude" config key maps to "anthropic" [FACT] (kosmos/core/providers/factory.py:83-175)
- Provider registry: `anthropic`, `claude` (alias), `openai`, `litellm`, `ollama`/`deepseek`/`lmstudio` (aliases for LiteLLMProvider) [FACT] (kosmos/core/providers/factory.py:189-217)

#### ClaudeClient (Legacy Path)
- `ClaudeClient.__init__(model=None, max_tokens=4096, temperature=0.7, enable_cache=True, enable_auto_model_selection=False)` -- Wraps Anthropic SDK; requires anthropic package and API key; CLI mode detected by all-9s key; optionally initializes ClaudeCache [FACT] (kosmos/core/llm.py:108-205)
- `ClaudeClient.generate(prompt, system=None, max_tokens=None, temperature=None, stop_sequences=None, model_override=None, bypass_cache=False) -> str` -- Single-prompt generation with cache, auto model selection (haiku vs sonnet), and usage tracking; returns content string [FACT] (kosmos/core/llm.py:237-365)
- `ClaudeClient.generate_structured(prompt, output_schema=None, schema=None, system=None, max_tokens=None, temperature=0.3, max_retries=2) -> Dict` -- JSON generation with retry on parse failure; bypasses cache on retries; exhausted retries raise ProviderAPIError(recoverable=False) [FACT] (kosmos/core/llm.py:416-486)
- `ClaudeClient.generate_with_messages(messages: List[Dict], system=None, max_tokens=None, temperature=None) -> str` -- Multi-turn conversation; no caching, no auto model selection [FACT] (kosmos/core/llm.py:367-404)
- `get_client(reset=False, use_provider_system=True) -> Union[ClaudeClient, LLMProvider]` -- Thread-safe singleton factory using double-checked locking; falls back to AnthropicProvider on config failure [FACT] (kosmos/core/llm.py:613-673)
- `get_provider() -> LLMProvider` -- Returns get_client() cast to LLMProvider; raises TypeError if result is not LLMProvider [FACT] (kosmos/core/llm.py:700-706)
- `ModelComplexity.estimate_complexity(prompt, system) -> dict` -- Heuristic 0-100 score; returns recommendation "haiku" (< 30) or "sonnet" (>= 30) with complexity_score, token estimate, keyword matches [FACT] (kosmos/core/llm.py:60-105)

### Workflow State Machine

#### WorkflowState and NextAction Enums
- `WorkflowState(str, Enum)` -- 9 states: INITIALIZING, GENERATING_HYPOTHESES, DESIGNING_EXPERIMENTS, EXECUTING, ANALYZING, REFINING, CONVERGED, PAUSED, ERROR [FACT] (kosmos/core/workflow.py:19-29)
- `NextAction(str, Enum)` -- 8 actions: GENERATE_HYPOTHESIS, DESIGN_EXPERIMENT, EXECUTE_EXPERIMENT, ANALYZE_RESULT, REFINE_HYPOTHESIS, CONVERGE, PAUSE, ERROR_RECOVERY [FACT] (kosmos/core/workflow.py:33-42)

#### ResearchPlan (BaseModel)
- `ResearchPlan.add_hypothesis(id)` -- Deduplicates before appending to hypothesis_pool [FACT] (kosmos/core/workflow.py:100-104)
- `ResearchPlan.mark_tested(id)` -- Adds to tested_hypotheses with dedup [FACT] (kosmos/core/workflow.py:106-109)
- `ResearchPlan.mark_supported(id)` -- Adds to supported_hypotheses AND implicitly calls mark_tested() [FACT] (kosmos/core/workflow.py:111-116)
- `ResearchPlan.mark_rejected(id)` -- Adds to rejected_hypotheses AND implicitly calls mark_tested() [FACT] (kosmos/core/workflow.py:118-123)
- `ResearchPlan.add_experiment(id)` -- Deduplicates into experiment_queue [FACT] (kosmos/core/workflow.py:125-128)
- `ResearchPlan.mark_experiment_complete(id)` -- Removes from experiment_queue and adds to completed_experiments [FACT] (kosmos/core/workflow.py:130-136)
- `ResearchPlan.add_result(id)` -- Deduplicates into results list [FACT] (kosmos/core/workflow.py:138-141)
- `ResearchPlan.get_untested_hypotheses() -> List[str]` -- Returns hypotheses in pool but not in tested list [FACT] (kosmos/core/workflow.py:149-151)
- `ResearchPlan.get_testability_rate() -> float` -- len(tested)/len(pool); returns 0.0 for empty pool [FACT] (kosmos/core/workflow.py:153-157)
- `ResearchPlan.get_support_rate() -> float` -- len(supported)/len(tested); returns 0.0 for empty tested [FACT] (kosmos/core/workflow.py:159-163)

#### ResearchWorkflow (State Machine Engine)
- `ResearchWorkflow.__init__(initial_state=WorkflowState.INITIALIZING, research_plan=None)` -- Initializes state machine with allowed transition table and empty history [FACT] (kosmos/core/workflow.py:229-243)
- `ResearchWorkflow.transition_to(new_state, action="", metadata=None) -> bool` -- Validates against ALLOWED_TRANSITIONS, records WorkflowTransition, synchronizes research_plan.current_state; raises ValueError for invalid transitions [FACT] (kosmos/core/workflow.py:260-328)
- `ResearchWorkflow.can_transition_to(target_state) -> bool` -- Checks if transition is allowed without actually transitioning [FACT] (kosmos/core/workflow.py:247-258)
- `ResearchWorkflow.reset()` -- Resets to INITIALIZING and clears all transition history permanently [FACT] (kosmos/core/workflow.py:342-349)
- `ResearchWorkflow.to_dict() -> dict` -- Exports current state, transition count, and last 5 transitions [FACT] (kosmos/core/workflow.py:351-365)
- `ResearchWorkflow.get_state_duration(state) -> float` -- Calculates total seconds spent in a given state by scanning transition history [FACT] (kosmos/core/workflow.py:367-396)
- `ResearchWorkflow.get_state_statistics() -> dict` -- Aggregates visit counts and durations for all visited states [FACT] (kosmos/core/workflow.py:398-416)

### Hypothesis Models

#### Hypothesis (BaseModel)
- `Hypothesis` -- Pydantic model with required fields: research_question, statement (10-500 chars), rationale (min 20 chars), domain; optional scores: testability_score, novelty_score, confidence_score, priority_score (all 0.0-1.0); evolution tracking via parent_hypothesis_id, generation, refinement_count [FACT] (kosmos/models/hypothesis.py:32-156)
- `Hypothesis.to_dict() -> dict` -- Manual serialization converting enums to .value and datetimes to .isoformat(); differs from model_dump() because use_enum_values=False [FACT] (kosmos/models/hypothesis.py:117-142)
- `Hypothesis.is_testable(threshold=0.3) -> bool` -- Returns False if testability_score is None [FACT] (kosmos/models/hypothesis.py:144-148)
- `Hypothesis.is_novel(threshold=0.5) -> bool` -- Returns False if novelty_score is None [FACT] (kosmos/models/hypothesis.py:150-154)
- `HypothesisStatus(str, Enum)` -- 6 values: GENERATED, UNDER_REVIEW, TESTING, SUPPORTED, REJECTED, INCONCLUSIVE [FACT] (kosmos/models/hypothesis.py:22-29)
- `ExperimentType(str, Enum)` -- 3 values: COMPUTATIONAL, DATA_ANALYSIS, LITERATURE_SYNTHESIS [FACT] (kosmos/models/hypothesis.py:15-19)

#### HypothesisGenerationRequest (BaseModel)
- `HypothesisGenerationRequest` -- Required: research_question (min 10 chars); defaults: num_hypotheses=3, max_iterations=1, require_novelty_check=True, min_novelty_score=0.5 [FACT] (kosmos/models/hypothesis.py:159-196)

#### HypothesisGenerationResponse (BaseModel)
- `HypothesisGenerationResponse.get_best_hypothesis() -> Optional[Hypothesis]` -- Returns highest priority_score hypothesis; silently returns first hypothesis as fallback when none scored [FACT] (kosmos/models/hypothesis.py:218-229)
- `HypothesisGenerationResponse.filter_testable(threshold=0.3) -> List[Hypothesis]` -- Filters by testability_score [FACT] (kosmos/models/hypothesis.py:231-233)
- `HypothesisGenerationResponse.filter_novel(threshold=0.5) -> List[Hypothesis]` -- Filters by novelty_score [FACT] (kosmos/models/hypothesis.py:235-237)

#### NoveltyReport (BaseModel)
- `NoveltyReport` -- Fields: novelty_score (0-1), max_similarity, prior_art_detected, is_novel, novelty_threshold_used=0.75; note threshold default differs from HypothesisGenerationRequest.min_novelty_score=0.5 [FACT] (kosmos/models/hypothesis.py:240-263)

#### TestabilityReport (BaseModel)
- `TestabilityReport` -- Fields: testability_score (0-1), is_testable, testability_threshold_used=0.3, primary_experiment_type, resource estimates, challenges, limitations [FACT] (kosmos/models/hypothesis.py:265-296)

#### PrioritizedHypothesis (BaseModel)
- `PrioritizedHypothesis.update_hypothesis_priority()` -- Mutates embedded hypothesis.priority_score and hypothesis.updated_at using weights: novelty=0.30, feasibility=0.25, impact=0.25, testability=0.20 [FACT] (kosmos/models/hypothesis.py:328-331)

### Experiment Models

#### ExperimentProtocol (BaseModel)
- `ExperimentProtocol` -- Central experiment model with required fields: name (min 5), hypothesis_id, experiment_type, domain, description (min 20), objective (min 10), steps (min 1), variables, resource_requirements; sample_size bounded 1-100,000; step numbering must be contiguous 1..N [FACT] (kosmos/models/experiment.py:351-575)
- `ExperimentProtocol.to_dict() -> dict` -- Manual 100-line serialization converting enums to .value; differs from model_dump() due to use_enum_values=False [FACT] (kosmos/models/experiment.py:471-573)
- `ExperimentProtocol.get_step(step_number) -> Optional[ProtocolStep]` -- Retrieves step by number [FACT] (kosmos/models/experiment.py:440)
- `ExperimentProtocol.get_independent_variables() -> List[Variable]` -- Filters variables by INDEPENDENT type [FACT] (kosmos/models/experiment.py:447)
- `ExperimentProtocol.get_dependent_variables() -> List[Variable]` -- Filters variables by DEPENDENT type [FACT] (kosmos/models/experiment.py:451)
- `ExperimentProtocol.has_control_group() -> bool` -- Checks if control_groups list is non-empty [FACT] (kosmos/models/experiment.py:455)
- `ExperimentProtocol.total_duration_estimate_days() -> float` -- Prefers resource_requirements.estimated_duration_days, falls back to sum of step durations [FACT] (kosmos/models/experiment.py:459-469)

#### ExperimentDesignRequest (BaseModel)
- `ExperimentDesignRequest` -- Input to experiment designer with hypothesis_id, preferred_experiment_type, max_cost_usd, max_duration_days, require_control_group (default True), require_power_analysis (default True), min_rigor_score (default 0.6) [FACT] (kosmos/models/experiment.py:578-613)

#### ExperimentDesignResponse (BaseModel)
- `ExperimentDesignResponse.is_feasible(max_cost=None, max_duration=None) -> bool` -- Checks cost and duration against constraints AND validation_passed [FACT] (kosmos/models/experiment.py:648-654)

#### Supporting Experiment Models
- `Variable(name, type: VariableType, description, values=None, fixed_value=None, unit=None, measurement_method=None)` -- Variable definition with min 10 char description [FACT] (kosmos/models/experiment.py:55-78)
- `ControlGroup(name, description, variables: Dict, rationale, sample_size=None)` -- Control group with coerce_sample_size validator that converts strings to ints and clamps to 100,000 [FACT] (kosmos/models/experiment.py:80-125)
- `ProtocolStep(step_number, title, description, action, requires_steps=[], expected_duration_minutes=None, code_template=None)` -- Protocol step with ensure_title replacing empty titles with "Untitled Step" silently [FACT] (kosmos/models/experiment.py:128-181)
- `ResourceRequirements(compute_hours, memory_gb, gpu_required=False, estimated_cost_usd, api_calls_estimated, required_libraries, can_parallelize=False)` -- Resource requirements with ge=0 constraints on all numeric fields; no cross-field validation [FACT] (kosmos/models/experiment.py:184-232)
- `StatisticalTestSpec(test_type, description, null_hypothesis, alternative="two-sided", alpha=0.05, required_power=0.8)` -- Statistical test specification with coerce_groups splitting comma strings and parse_effect_size extracting floats via regex [FACT] (kosmos/models/experiment.py:235-301)
- `VariableType(Enum)` -- INDEPENDENT, DEPENDENT, CONTROL, CONFOUNDING [FACT] (kosmos/models/experiment.py:21-26)
- `StatisticalTest(Enum)` -- T_TEST, ANOVA, CHI_SQUARE, CORRELATION, REGRESSION, MANN_WHITNEY, KRUSKAL_WALLIS, WILCOXON, CUSTOM [FACT] (kosmos/models/experiment.py:29-39)
- `ValidationReport(BaseModel)` -- Protocol validation with rigor_score, checks, control group adequacy, sample size adequacy, power analysis, bias detection, reproducibility scoring [FACT] (kosmos/models/experiment.py:657-699)

### Result Models

#### ExperimentResult (BaseModel)
- `ExperimentResult` -- Central result model with required: experiment_id, protocol_id, status (ResultStatus), metadata (ExecutionMetadata); optional: raw_data, processed_data, variable_results, statistical_tests, primary_test, supports_hypothesis (tri-state: True/False/None), version (default 1) [FACT] (kosmos/models/result.py:127-204)
- `ExperimentResult.get_primary_test_result() -> Optional[StatisticalTestResult]` -- Linear scan for the primary test; returns None if not set [FACT] (kosmos/models/result.py:230-239)
- `ExperimentResult.to_dict() -> dict` -- Uses model_to_dict compat layer (unlike manual to_dict in other models) [FACT] (kosmos/models/result.py:241-243)
- `ExperimentResult.to_json() -> str` -- Uses Pydantic v2-native model_dump_json [FACT] (kosmos/models/result.py:245-247)
- `ExperimentResult.from_dict(data) -> ExperimentResult` -- Class method using model_validate [FACT] (kosmos/models/result.py:249-251)
- `ExperimentResult.from_json(json_str) -> ExperimentResult` -- Class method using model_validate_json [FACT] (kosmos/models/result.py:253-257)
- `ExperimentResult.is_significant(alpha=0.05) -> bool` -- Checks primary_p_value < alpha; returns False if no p-value [FACT] (kosmos/models/result.py:259-263)
- `ExperimentResult.get_summary_stats() -> dict` -- Aggregates variable stats keyed by variable name [FACT] (kosmos/models/result.py:265-277)
- `ResultStatus(str, Enum)` -- 5 values: SUCCESS, FAILED, PARTIAL, TIMEOUT, ERROR; FAILED means experiment ran but no useful results, ERROR means runtime/system error [FACT] (kosmos/models/result.py:17-22)

#### Supporting Result Models
- `ExecutionMetadata(BaseModel)` -- Execution environment: start_time, end_time, duration_seconds, python_version, platform, experiment_id, protocol_id, sandbox_used, timeout_occurred [FACT] (kosmos/models/result.py:25-58)
- `StatisticalTestResult(BaseModel)` -- Test output with statistic, p_value (0-1), effect_size, confidence_interval, three significance flags at alpha 0.05/0.01/0.001 (all required), significance_label (***/**/*/ns) [FACT] (kosmos/models/result.py:61-103)
- `VariableResult(BaseModel)` -- Summary stats for a variable: mean, median, std, min, max, optional values list, n_samples, n_missing [FACT] (kosmos/models/result.py:105-124)
- `ResultExport(result, format)` -- Export wrapper supporting JSON, CSV (lazy pandas import), and Markdown formats [FACT] (kosmos/models/result.py:280-377)
- `ResultExport.export_csv() -> str` -- Imports pandas at call time; creates DataFrame from variable_results [FACT] (kosmos/models/result.py:292-310)
- `ResultExport.export_markdown() -> str` -- Generates structured markdown report; will crash on None stats due to :.2f formatting [FACT] (kosmos/models/result.py:312-377)

### Code Execution

#### CodeExecutor
- `CodeExecutor.__init__(max_retries=3, retry_delay=1, allowed_globals=None, use_sandbox=True, sandbox_config=None, enable_profiling=False, execution_timeout=300)` -- Initializes execution engine; if Docker unavailable, use_sandbox silently set to False [FACT] (kosmos/execution/executor.py:174-235)
- `CodeExecutor.execute(code, local_vars=None, llm_client=None) -> ExecutionResult` -- Main entry point; auto-detects language (Python/R), enters retry loop with code mutation via RetryStrategy; all exceptions caught, final failure returns error with type "MaxRetriesExceeded" [FACT] (kosmos/execution/executor.py:237-376)
- `CodeExecutor.execute_with_data(code, data_path, local_vars=None) -> ExecutionResult` -- Data_path injected both as prepended code assignment and local variable (belt-and-suspenders, Issue #51) [FACT] (kosmos/execution/executor.py:632-660)
- `CodeExecutor.execute_r(code, local_vars=None) -> ExecutionResult` -- R code execution wrapper converting RExecutionResult to ExecutionResult [FACT] (kosmos/execution/executor.py:378-464)
- `CodeExecutor.is_r_available() -> bool` -- Checks if R executor is available at runtime [FACT] (kosmos/execution/executor.py:378-464)
- `ExecutionResult` -- Data class with success, return_value, stdout, stderr, error_type, error_message, execution_time, profile_data, data_source ('file' or 'synthetic') [FACT] (kosmos/execution/executor.py:113-159)
- `ExecutionResult.to_dict() -> dict` -- Serializes; profile data wrapped in try/except for non-serializable objects [FACT] (kosmos/execution/executor.py:153-156)

#### RetryStrategy (Code Repair Engine)
- `RetryStrategy.modify_code_for_retry(code, error_type, error_message, attempt, llm_client=None) -> Optional[str]` -- Core repair method; LLM repair first (limited to first 2 attempts), then pattern-based fixes for 10+ error types; FileNotFoundError returns None (terminal); wraps other errors in try/except [FACT] (kosmos/execution/executor.py:751-825)
- `RetryStrategy.should_retry(error_type, error_message) -> bool` -- SyntaxError, FileNotFoundError, DataUnavailableError are explicitly non-retryable [FACT] (kosmos/execution/executor.py:717-730)

#### execute_protocol_code()
- `execute_protocol_code(code, local_vars=None, use_sandbox=True, llm_client=None) -> ExecutionResult` -- Convenience function; code is ALWAYS validated via CodeValidator before execution (removed safety bypass); validation failure returns immediately without executing [FACT] (kosmos/execution/executor.py:1017-1067)

### Safety

#### SafetyGuardrails
- `SafetyGuardrails.__init__(incident_log_path="safety_incidents.jsonl", enable_signal_handlers=True, docker_client=None)` -- Initializes CodeValidator with hardcoded permissions (allow_file_read=True, allow_file_write=False, allow_network=False); registers SIGTERM/SIGINT handlers; default resource limits are deny-by-default [FACT] (kosmos/safety/guardrails.py:46-93)
- `SafetyGuardrails.validate_code(code, context=None) -> SafetyReport` -- Raises RuntimeError if emergency stop active; validates code, logs violations as incidents; returns report for caller to decide [FACT] (kosmos/safety/guardrails.py:112-140)
- `SafetyGuardrails.enforce_resource_limits(requested: ResourceLimit) -> ResourceLimit` -- Caps requested limits using min() for numerics and AND for booleans; since defaults are all False, boolean permissions are always blocked [FACT] (kosmos/safety/guardrails.py:142-178)
- `SafetyGuardrails.check_emergency_stop()` -- Dual-source check: flag file AND in-memory status; raises RuntimeError if stop active [FACT] (kosmos/safety/guardrails.py:180-203)
- `SafetyGuardrails.is_emergency_stop_active() -> bool` -- Has write side effect: if flag file exists and stop not already active, TRIGGERS the stop; read operation that can cause state change [FACT] (kosmos/safety/guardrails.py:205-214)
- `SafetyGuardrails.trigger_emergency_stop(reason, triggered_by)` -- Sets in-memory status, creates flag file with JSON, kills all Docker containers with kosmos.sandbox=true label, logs incident [FACT] (kosmos/safety/guardrails.py:216-288)
- `SafetyGuardrails.reset_emergency_stop()` -- Resets in-memory status and removes flag file; if file removal fails, next is_emergency_stop_active() call will re-trigger [FACT] (kosmos/safety/guardrails.py:290-302)
- `SafetyGuardrails.safety_context()` -- Context manager wrapping code execution with pre-execution emergency stop check, exception monitoring, and post-execution check [FACT] (kosmos/safety/guardrails.py:304-345)
- `SafetyGuardrails.get_recent_incidents(limit=10, min_severity=None) -> List[SafetyIncident]` -- Returns last N incidents; emergency stop incidents (violation=None) excluded when filtering by severity [FACT] (kosmos/safety/guardrails.py:382-407)
- `SafetyGuardrails.get_incident_summary() -> dict` -- Aggregate statistics; emergency stop count uses fragile string matching on incident_id prefix [FACT] (kosmos/safety/guardrails.py:409-452)

### Monitoring and Alerts

#### AlertManager
- `AlertManager.__init__()` -- Initializes with 7 default rules (3 are placeholders that never fire); no thread safety [FACT] (kosmos/monitoring/alerts.py:125-135)
- `AlertManager.add_alert_rule(rule: AlertRule)` -- Appends rule; no dedup check, same name can be added multiple times [FACT] (kosmos/monitoring/alerts.py:203-206)
- `AlertManager.add_notification_handler(handler: Callable)` -- Registers notification handler; lambdas log as "<lambda>" [FACT] (kosmos/monitoring/alerts.py:208-216)
- `AlertManager.evaluate_rules()` -- Iterates all rules synchronously, triggers alerts where conditions are met [FACT] (kosmos/monitoring/alerts.py:218-222)
- `AlertManager.resolve_alert(alert_id)` -- Marks RESOLVED and removes from active_alerts; no-op if not found [FACT] (kosmos/monitoring/alerts.py:249-255)
- `AlertManager.acknowledge_alert(alert_id)` -- Marks ACKNOWLEDGED but does NOT remove from active_alerts [FACT] (kosmos/monitoring/alerts.py:257-262)
- `AlertManager.get_active_alerts() -> List[Alert]` -- Returns list copy of active alert values [FACT] (kosmos/monitoring/alerts.py:264-266)
- `AlertManager.get_alert_history(limit=100) -> List[Alert]` -- Returns last N alerts from history [FACT] (kosmos/monitoring/alerts.py:268-270)
- `Alert(name, severity, message, timestamp, status=ACTIVE, details={}, alert_id=None)` -- Dataclass; auto-generates alert_id as "{name}_{unix_timestamp}"; same-second same-name alerts collide [FACT] (kosmos/monitoring/alerts.py:34-49)
- `AlertRule(name, condition: Callable, severity, message_template, cooldown_seconds=300)` -- Trigger condition; should_trigger() catches condition exceptions and returns False (broken checker treated as "no alert") [FACT] (kosmos/monitoring/alerts.py:64-93)
- `AlertSeverity(str, Enum)` -- INFO, WARNING, ERROR, CRITICAL [FACT] (kosmos/monitoring/alerts.py:19-24)
- `AlertStatus(str, Enum)` -- ACTIVE, RESOLVED, ACKNOWLEDGED [FACT] (kosmos/monitoring/alerts.py:27-31)

#### Module-Level Alert Functions
- `get_alert_manager() -> AlertManager` -- Singleton; registers log handler unconditionally, email/Slack/PagerDuty handlers from env vars at creation time only [FACT] (kosmos/monitoring/alerts.py:528-552)
- `evaluate_alerts()` -- Convenience wrapper for get_alert_manager().evaluate_rules() [FACT] (kosmos/monitoring/alerts.py:555-557)
- `log_notification_handler(alert)` -- Maps severity to logger method; always on [FACT] (kosmos/monitoring/alerts.py:337-346)
- `slack_notification_handler(alert)` -- Sends Slack webhook with color-coded attachments; gated by ALERT_SLACK_ENABLED env var [FACT] (kosmos/monitoring/alerts.py:407-472)
- `pagerduty_notification_handler(alert)` -- Posts to PagerDuty Events API v2; silently drops INFO and WARNING (only ERROR and CRITICAL) [FACT] (kosmos/monitoring/alerts.py:475-511)

### Knowledge Graph

#### KnowledgeGraph
- `KnowledgeGraph.__init__(uri=None, user=None, password=None, database=None, auto_start_container=True, create_indexes=True)` -- Neo4j connection via py2neo; constructor may trigger Docker operations to start container; silent connection failure sets graph=None [FACT] (kosmos/knowledge/graph.py:35-111)
- `KnowledgeGraph.create_paper(paper: PaperMetadata, merge=True) -> Node` -- Merge-or-create pattern with read-then-write for existing papers [FACT] (kosmos/knowledge/graph.py:202-260)
- `KnowledgeGraph.get_paper(paper_id) -> Optional[Node]` -- Cascades through 4 separate queries: primary id, DOI, arxiv_id, pubmed_id [FACT] (kosmos/knowledge/graph.py:262-289)
- `KnowledgeGraph.delete_paper(paper_id)` -- Cascading delete: removes node AND all its relationships [FACT] (kosmos/knowledge/graph.py:323)
- `KnowledgeGraph.create_author(name, affiliation=None, h_index=None, merge=True) -> Node` -- Authors identified by name string (fragile for disambiguation) [FACT] (kosmos/knowledge/graph.py:332-401)
- `KnowledgeGraph.create_concept(name, domain=None, merge=True) -> Node` -- Same merge pattern [FACT] (kosmos/knowledge/graph.py:403-469)
- `KnowledgeGraph.create_method(name, category=None, merge=True) -> Node` -- Same merge pattern [FACT] (kosmos/knowledge/graph.py:471-537)
- `KnowledgeGraph.create_citation(citing_paper_id, cited_paper_id, merge=True) -> Relationship` -- CITES relationship between two papers [FACT] (kosmos/knowledge/graph.py:541-575)
- `KnowledgeGraph.create_authored(paper_id, author_name, merge=True) -> Relationship` -- AUTHORED relationship; non-idempotent: increments author.paper_count on every call even if relationship already exists [FACT] (kosmos/knowledge/graph.py:577-618)
- `KnowledgeGraph.create_discusses(paper_id, concept_name, merge=True) -> Relationship` -- DISCUSSES relationship; same counter bug: concept.frequency incremented unconditionally [FACT] (kosmos/knowledge/graph.py:620-662)
- `KnowledgeGraph.create_uses_method(paper_id, method_name, merge=True) -> Relationship` -- USES_METHOD relationship; same counter bug: method.usage_count incremented unconditionally [FACT] (kosmos/knowledge/graph.py:664-706)
- `KnowledgeGraph.get_citations(paper_id, depth=1) -> List` -- Variable-depth citation traversal [FACT] (kosmos/knowledge/graph.py:749-767)
- `KnowledgeGraph.find_related_papers(paper_id, max_hops=2) -> List` -- Multi-hop graph traversal [FACT] (kosmos/knowledge/graph.py:898-934)
- `KnowledgeGraph.get_concept_cooccurrence() -> List` -- Finds concepts co-appearing in papers [FACT] (kosmos/knowledge/graph.py:936-966)
- `KnowledgeGraph.get_stats() -> dict` -- Runs 9 separate Cypher queries (4 node types + 5 relationship types) [FACT] (kosmos/knowledge/graph.py:968-989)
- `KnowledgeGraph.clear_graph()` -- MATCH (n) DETACH DELETE n: deletes ALL data with no confirmation [FACT] (kosmos/knowledge/graph.py:995)
- `get_knowledge_graph() -> KnowledgeGraph` -- Singleton accessor; NOT thread-safe on first access [FACT] (kosmos/knowledge/graph.py:999-1038)

### World Model

#### Neo4jWorldModel (Adapter)
- `Neo4jWorldModel` -- Implements WorldModelStorage + EntityManager interfaces by wrapping KnowledgeGraph singleton; type-dispatch routes Paper/Concept/Author/Method to existing graph methods, custom types use generic Cypher [FACT] (kosmos/world_model/simple.py:45-89)
- `Neo4jWorldModel.add_entity(entity: Entity) -> str` -- Dispatches by entity.type to type-specific private methods; returns entity.id silently without persisting if Neo4j disconnected [FACT] (kosmos/world_model/simple.py:92-128)
- `Neo4jWorldModel.get_entity(entity_id) -> Optional[Entity]` -- Tries get_paper() first (4 DB queries), then generic Cypher; returns None if not found [FACT] (kosmos/world_model/simple.py:274-320)
- `Neo4jWorldModel.update_entity(entity_id, updates: Dict)` -- Cypher SET n += $updates merge-update; raises ValueError if not found (inconsistent with get_entity returning None) [FACT] (kosmos/world_model/simple.py:395-427)
- `Neo4jWorldModel.delete_entity(entity_id)` -- DETACH DELETE removing node and all relationships; raises ValueError if not found [FACT] (kosmos/world_model/simple.py:429-456)
- `Neo4jWorldModel.add_relationship(relationship: Relationship)` -- Routes CITES to create_citation, AUTHOR_OF to create_authored, others to generic Cypher [FACT] (kosmos/world_model/simple.py:458-500)
- `Neo4jWorldModel.query_related_entities(entity_id, relationship_type, direction, max_depth) -> List[Entity]` -- Graph traversal; results limited to 100 entities (hardcoded); direction accepts "outgoing"/"incoming"/"both" with no validation [FACT] (kosmos/world_model/simple.py:596-647)
- `Neo4jWorldModel.export_graph(filepath, project=None) -> dict` -- Full JSON export; writes empty file if Neo4j disconnected (potential data loss) [FACT] (kosmos/world_model/simple.py:649-756)
- `Neo4jWorldModel.import_graph(filepath, clear_existing=False)` -- JSON import with per-relationship error handling; partial imports can occur silently [FACT] (kosmos/world_model/simple.py:758-826)
- `Neo4jWorldModel.get_statistics(project=None) -> dict` -- 5 separate Cypher queries for counts, types, projects, storage size [FACT] (kosmos/world_model/simple.py:828-901)
- `Neo4jWorldModel.reset()` -- Calls graph.clear_graph() which deletes ALL data; destructive, no confirmation [FACT] (kosmos/world_model/simple.py:963-982)
- `Neo4jWorldModel.verify_entity(entity_id, verified_by)` -- Sets verified=true on node; raises ValueError if not found [FACT] (kosmos/world_model/simple.py:995-1024)
- `Neo4jWorldModel.add_annotation(entity_id, annotation: Annotation)` -- Appends to JSON array on node; uses deprecated datetime.utcnow() [FACT] (kosmos/world_model/simple.py:1026-1086)
- `Neo4jWorldModel.get_annotations(entity_id) -> List[Annotation]` -- Deserializes annotation array; malformed entries skipped with warnings [FACT] (kosmos/world_model/simple.py:1088-1159)

### Literature Clients

#### BaseLiteratureClient (Abstract)
- `BaseLiteratureClient.__init__(api_key=None, cache_enabled=True)` -- Stores API key and cache flag; creates namespaced logger [FACT] (kosmos/literature/base_client.py:133-143)
- `BaseLiteratureClient.search(query, max_results=10, fields=None, year_from=None, year_to=None, **kwargs) -> List[PaperMetadata]` -- Abstract; returns papers matching query [FACT] (kosmos/literature/base_client.py:145-169)
- `BaseLiteratureClient.get_paper_by_id(paper_id) -> Optional[PaperMetadata]` -- Abstract; single paper lookup [FACT] (kosmos/literature/base_client.py:171-182)
- `BaseLiteratureClient.get_paper_references(paper_id, max_refs=50) -> List[PaperMetadata]` -- Abstract; papers cited by the given paper [FACT] (kosmos/literature/base_client.py:184-196)
- `BaseLiteratureClient.get_paper_citations(paper_id, max_cites=50) -> List[PaperMetadata]` -- Abstract; papers citing the given paper [FACT] (kosmos/literature/base_client.py:198-210)
- `BaseLiteratureClient.get_source_name() -> str` -- Returns class name with "Client" removed [FACT] (kosmos/literature/base_client.py:212-219)
- `BaseLiteratureClient._validate_query(query) -> bool` -- Returns False for empty queries; logs "truncating" for >1000 chars but does NOT truncate [FACT] (kosmos/literature/base_client.py:235-253)
- `BaseLiteratureClient._normalize_paper_metadata(raw_data) -> PaperMetadata` -- Raises NotImplementedError (not @abstractmethod); subclass can be instantiated without overriding [FACT] (kosmos/literature/base_client.py:255-268)

#### PaperMetadata (dataclass)
- `PaperMetadata` -- Unified paper representation with id, source (PaperSource), doi, arxiv_id, pubmed_id, title, abstract, authors (List[Author]), publication_date, citation_count, fields, keywords, raw_data [FACT] (kosmos/literature/base_client.py:36-78)
- `PaperMetadata.primary_identifier` -- Property returning DOI > arXiv > PubMed > source ID priority chain [FACT] (kosmos/literature/base_client.py:89-92)
- `PaperMetadata.author_names -> List[str]` -- Property extracting name strings from Author objects [FACT] (kosmos/literature/base_client.py:94-97)
- `PaperMetadata.to_dict() -> dict` -- Hand-written serialization; raw_data excluded; authors converted to dicts [FACT] (kosmos/literature/base_client.py:99-122)
- `Author(name, affiliation=None, email=None, author_id=None)` -- Plain dataclass with no validation [FACT] (kosmos/literature/base_client.py:27-32)
- `PaperSource(str, Enum)` -- ARXIV, SEMANTIC_SCHOLAR, PUBMED, UNKNOWN, MANUAL [FACT] (kosmos/literature/base_client.py:17-24)

### Configuration

#### KosmosConfig (Pydantic BaseSettings)
- `get_config(reload=False) -> KosmosConfig` -- Global singleton; on first call instantiates from environment/.env and creates log/ChromaDB directories on disk; raises ValidationError if required fields missing [FACT] (kosmos/config.py:1140-1154)
- `reset_config()` -- Sets singleton to None, forcing re-creation on next get_config() [FACT] (kosmos/config.py:1157-1160)
- `KosmosConfig.get_active_model() -> str` -- Returns model string for currently active LLM provider; raises ValueError for unknown provider [FACT] (kosmos/config.py:1045-1054)
- `KosmosConfig.get_active_provider_config() -> dict` -- Returns dict with model, api_key, and optionally api_base; raises ValueError/AttributeError if provider config is None [FACT] (kosmos/config.py:1056-1065)
- `KosmosConfig.validate_dependencies() -> List[str]` -- Returns list of missing dependency descriptions [FACT] (kosmos/config.py:1078-1101)
- `KosmosConfig.to_dict() -> dict` -- Full config serialization via model_to_dict [FACT] (kosmos/config.py:1103-1133)
- `ClaudeConfig.is_cli_mode -> bool` -- Returns True when API key consists entirely of digit 9 (any length); bypass for CLI mode [FACT] (kosmos/config.py:80-82)

#### Config Sections (16 nested BaseSettings)
- `ClaudeConfig` -- api_key, model, max_tokens, temperature, enable_cache; env prefix ANTHROPIC_API_KEY, CLAUDE_* [FACT] (kosmos/config.py:29-84)
- `OpenAIConfig` -- api_key, model, base_url; env prefix OPENAI_* [FACT] (kosmos/config.py:91-141)
- `LiteLLMConfig` -- model, api_key, api_base; env prefix LITELLM_* [FACT] (kosmos/config.py:143-197)
- `ResearchConfig` -- max_iterations, enabled_domains, budget_usd, max_runtime_hours (upper bound 24.0) [FACT] (kosmos/config.py:200-247)
- `DatabaseConfig` -- url, echo; normalized_url property resolves relative SQLite paths to absolute [FACT] (kosmos/config.py:250-302)
- `SafetyConfig` -- enable_safety_checks, enable_sandboxing, approval_mode [FACT] (kosmos/config.py:573-683)
- `PerformanceConfig` -- enable_concurrent_operations, max_concurrent_experiments [FACT] (kosmos/config.py:686-749)
- `WorldModelConfig` -- enabled, mode, project [FACT] (kosmos/config.py:865-893)

### Database

#### Session Management
- `init_database(database_url, echo=False, pool_size=5, max_overflow=10, pool_timeout=30, enable_slow_query_logging=True, slow_query_threshold_ms=100.0)` -- Creates SQLAlchemy engine with database-type-specific config (SQLite: StaticPool/check_same_thread=False; PostgreSQL: QueuePool/pool_pre_ping=True); calls Base.metadata.create_all() [FACT] (kosmos/db/__init__.py:26-100)
- `get_session() -> ContextManager[Session]` -- Yields SQLAlchemy Session; auto-commits on clean exit, auto-rolls back on exception; raises RuntimeError if database not initialized [FACT] (kosmos/db/__init__.py:108-137)
- `init_from_config()` -- High-level initialization from KosmosConfig; runs first_time_setup(), init_database(), verifies schema; degraded operation on setup errors [FACT] (kosmos/db/__init__.py:140-188)
- `reset_database()` -- Drops ALL tables and recreates; destroys all data; testing/evaluation only [FACT] (kosmos/db/__init__.py:191-202)

#### CRUD Operations
- `create_hypothesis(session, research_question, statement, rationale, domain, ...) -> Hypothesis` -- Validates related_papers as list; commits and refreshes [FACT] (kosmos/db/operations.py:87-114)
- `get_hypothesis(session, hypothesis_id, with_experiments=True) -> Optional[Hypothesis]` -- By ID with optional joinedload of experiments [FACT] (kosmos/db/operations.py:135)
- `list_hypotheses(session, domain=None, status=None, limit=100) -> List[Hypothesis]` -- Filtered query; orders by created_at DESC [FACT] (kosmos/db/operations.py:143-172)
- `update_hypothesis_status(session, hypothesis_id, status) -> Hypothesis` -- Raises ValueError if missing; sets updated_at [FACT] (kosmos/db/operations.py:175-186)
- `create_experiment(session, id, hypothesis_id, experiment_type, description, protocol: Dict, domain, code_generated=None) -> Experiment` -- Validates protocol as required dict [FACT] (kosmos/db/operations.py:198-221)
- `get_experiment(session, experiment_id, with_hypothesis=True, with_results=False) -> Optional[Experiment]` -- By ID with configurable eager loading [FACT] (kosmos/db/operations.py:227-256)
- `update_experiment_status(session, experiment_id, status, error_message=None, execution_time_seconds=None)` -- Sets started_at on RUNNING, completed_at on COMPLETED/FAILED; raises ValueError if not found [FACT] (kosmos/db/operations.py:308-327)
- `create_result(session, id, experiment_id, data: Dict, statistical_tests=None, key_findings=None, supports_hypothesis=None, p_value=None, effect_size=None) -> Result` -- Validates data as required dict, statistical_tests as optional dict, key_findings as optional list [FACT] (kosmos/db/operations.py:338-365)
- `get_result(session, result_id) -> Optional[Result]` -- By ID with optional joinedload of experiment [FACT] (kosmos/db/operations.py:389)
- `create_research_session(session, id, research_question, domain, max_iterations=10, autonomous_mode=True) -> ResearchSession` -- Creates tracking session [FACT] (kosmos/db/operations.py:539-558)
- `update_research_session(session, session_id, status=None, iteration=None, hypotheses_generated=None, experiments_completed=None)` -- Sets completed_at when status becomes "completed"; raises ValueError if not found [FACT] (kosmos/db/operations.py:572-594)

### Logging

#### Core Logging
- `setup_logging(level="INFO", log_format="text", log_file=None, debug_mode=False) -> logging.Logger` -- Configures root logger with console (stdout) and optional rotating file handler (10MB max, 5 backups); clears ALL existing root logger handlers first [FACT] (kosmos/core/logging.py:133-217)
- `get_logger(name) -> logging.Logger` -- Thin wrapper around logging.getLogger(name) [FACT] (kosmos/core/logging.py:220-239)
- `configure_from_config()` -- Reads logging settings from KosmosConfig singleton and calls setup_logging() [FACT] (kosmos/core/logging.py:383-403)
- `JSONFormatter.format(record) -> str` -- Serializes to JSON with UTC timestamp, level, correlation_id (if set via ContextVar), workflow context fields (if attached) [FACT] (kosmos/core/logging.py:34-82)
- `TextFormatter.format(record) -> str` -- Human-readable output with optional ANSI colors; mutates record.levelname in-place which can leak to other handlers [FACT] (kosmos/core/logging.py:85-130)
- `ExperimentLogger` -- Tracks experiment lifecycle (start, hypothesis, design, execution, result, error, end) via structured logging; all in-memory, no persistence [FACT] (kosmos/core/logging.py:242-380)
- `ExperimentLogger.start()` / `ExperimentLogger.end(status)` / `ExperimentLogger.get_summary() -> dict` -- Lifecycle methods; end() without prior start() returns duration 0 silently [FACT] (kosmos/core/logging.py:242-380)

### Experiment Caching

#### ExperimentCache
- `ExperimentCache.__init__(cache_dir=".kosmos_cache/experiments", similarity_threshold=0.90, enable_similarity=True, max_similar_results=5)` -- SQLite-backed cache; creates directory and database on init; thread-safe via RLock [FACT] (kosmos/core/experiment_cache.py:190-238)
- `ExperimentCache.cache_result(hypothesis, parameters, results, execution_time, metadata=None, embedding=None) -> str` -- Stores experiment with SHA256 fingerprint; INSERT OR REPLACE silently overwrites; returns experiment_id as "exp_{fingerprint[:16]}" [FACT] (kosmos/core/experiment_cache.py:302-393)
- `ExperimentCache.get_cached_result(hypothesis, parameters) -> Optional[ExperimentCacheEntry]` -- Exact match by fingerprint; returns most recent entry; increments hit/miss stats [FACT] (kosmos/core/experiment_cache.py:395-446)
- `ExperimentCache.find_similar(hypothesis, parameters, embedding=None) -> List[Tuple[ExperimentCacheEntry, float]]` -- Linear scan O(n) cosine similarity against all cached embeddings; no vector index [FACT] (kosmos/core/experiment_cache.py:448-522)
- `ExperimentCache.get_stats() -> dict` -- Cache statistics including hit rate, DB file size [FACT] (kosmos/core/experiment_cache.py:595-651)
- `ExperimentCache.clear() -> int` -- Deletes all cached data permanently; returns count [FACT] (kosmos/core/experiment_cache.py:653-687)
- `ExperimentCache.get_recent_experiments(limit=10) -> List[ExperimentCacheEntry]` -- Most recent by timestamp [FACT] (kosmos/core/experiment_cache.py:689-722)
- `get_experiment_cache(similarity_threshold=None, enable_similarity=None) -> ExperimentCache` -- Singleton; parameters only applied on first call, silently ignored on subsequent calls [FACT] (kosmos/core/experiment_cache.py:729-751)
- `ExperimentCacheEntry` -- Data container with experiment_id, hypothesis, parameters, results, execution_time, timestamp, metadata, embedding [FACT] (kosmos/core/experiment_cache.py:23-57)
- `ExperimentNormalizer.generate_fingerprint(hypothesis, parameters) -> str` -- Normalizes parameters, SHA256-hashes combined representation [FACT] (kosmos/core/experiment_cache.py:126-150)

### CLI

#### Entry Point
- `cli_entrypoint()` -- Installed entry point wrapping Typer app; catches KeyboardInterrupt (exit 130) and generic exceptions (exit 1); debug detection via raw sys.argv check [FACT] (kosmos/cli/main.py:422-440)
- `main(ctx, verbose, debug, trace, debug_level, debug_modules, quiet)` -- Typer callback running before any subcommand; initializes database (failure non-fatal for help/version), stores global options in ctx.obj [FACT] (kosmos/cli/main.py:98-170)
- `version()` -- Displays version, Python version, platform, Anthropic SDK version [FACT] (kosmos/cli/main.py:172-196)
- `info()` -- Displays system configuration, cache status, API key status [FACT] (kosmos/cli/main.py:199-263)
- `doctor()` -- Runs diagnostic checks: Python version, 8 required packages, API key, cache directory, database connection+schema [FACT] (kosmos/cli/main.py:266-392)
- `register_commands()` -- Registers 7 subcommands (run, status, history, cache, config, profile, graph); single ImportError disables ALL commands silently [FACT] (kosmos/cli/main.py:397-416)
## Data Contracts

This section catalogs every Pydantic model, dataclass, and structured return type that flows across module boundaries in Kosmos. Each entry documents the model's type system, key fields, serialization strategy, cross-boundary flow, and known gotchas. (see Critical Path 0 for Hypothesis/Experiment/Result flow through the research pipeline; see Module Index for behavioral details of each model)

### Pipeline Data Models (Hypothesis -> Experiment -> Result)

These three models form the core data pipeline. ID-based foreign keys link them: `ExperimentProtocol.hypothesis_id` [FACT: experiment.py:353] references `Hypothesis.id` [FACT: hypothesis.py:50]; `ExperimentResult.hypothesis_id` [FACT: result.py:138] references `Hypothesis.id`; `ExperimentResult.protocol_id` [FACT: result.py:137] references `ExperimentProtocol.id` [FACT: experiment.py:351]; `ExecutionMetadata.experiment_id` [FACT: result.py:49] and `protocol_id` [FACT: result.py:50] reference the same.

| Model | File | Type | Key Fields | Serialization | Cross-Boundary Flow | Gotcha |
|-------|------|------|-----------|---------------|--------------------|---------| 
| `Hypothesis` | `kosmos/models/hypothesis.py:32-156` | Pydantic BaseModel | `id: Optional[str]` [FACT: hypothesis.py:50], `statement: str` (10-500 chars) [FACT: hypothesis.py:52], `rationale: str` (min 20) [FACT: hypothesis.py:53], `domain: str` [FACT: hypothesis.py:55], `status: HypothesisStatus` [FACT: hypothesis.py:56], `testability_score: Optional[float]` [FACT: hypothesis.py:59], `novelty_score: Optional[float]` [FACT: hypothesis.py:60], `priority_score: Optional[float]` [FACT: hypothesis.py:62], `suggested_experiment_types: List[ExperimentType]` [FACT: hypothesis.py:65], `parent_hypothesis_id: Optional[str]` [FACT: hypothesis.py:81], `generation: int` [FACT: hypothesis.py:82] | Manual `to_dict()` [FACT: hypothesis.py:117-142]: converts enums via `.value`, datetimes via `.isoformat()`. Returns 20 keys. `model_dump()` returns enum objects (not strings) due to `use_enum_values=False` [FACT: hypothesis.py:156]. | hypothesis_generator -> research_director -> convergence, feedback, memory; FK target for ExperimentProtocol and ExperimentResult; Dual model: Pydantic runtime + SQLAlchemy `HypothesisModel` in `kosmos.db.models` [FACT: hypothesis module findings] | `to_dict()` and `model_dump()` produce DIFFERENT output [FACT: hypothesis module findings]. `None` scores fail `is_testable()` and `is_novel()` silently [FACT: hypothesis.py:146-147,152-153]. Three different novelty thresholds: 0.75, 0.5, 0.5 [FACT: hypothesis module findings]. `statement` validator permits non-predictive hypotheses (soft check only) [FACT: hypothesis.py:98-101]. `datetime.utcnow` deprecated in Python 3.12+ [FACT: hypothesis.py:76-77]. |
| `HypothesisStatus` | `kosmos/models/hypothesis.py:22-29` | str Enum | `GENERATED` [FACT: hypothesis.py:24], `UNDER_REVIEW` [FACT: hypothesis.py:25], `TESTING` [FACT: hypothesis.py:26], `SUPPORTED` [FACT: hypothesis.py:27], `REJECTED` [FACT: hypothesis.py:28], `INCONCLUSIVE` [FACT: hypothesis.py:29] | Stored as enum object in Hypothesis (not string) due to `use_enum_values=False` [FACT: hypothesis.py:156]. `to_dict()` calls `.value` explicitly [FACT: hypothesis.py:126]. | Used by convergence detector, research_director status tracking, feedback patterns | Removing any value breaks 48 importers and 15 tests |
| `ExperimentType` | `kosmos/models/hypothesis.py:15-19` | str Enum | `COMPUTATIONAL` [FACT: hypothesis.py:17], `DATA_ANALYSIS` [FACT: hypothesis.py:18], `LITERATURE_SYNTHESIS` [FACT: hypothesis.py:19] | `.value` serialization | Shared between hypothesis and experiment modules [FACT: experiment.py:14 imports it]. Determines template selection, code generation path, and resource estimation. | Only 3 values. Imported by `experiment.py` [FACT: experiment.py:14], bridging hypothesis-experiment boundary. |
| `HypothesisGenerationRequest` | `kosmos/models/hypothesis.py:159-197` | Pydantic BaseModel | `research_question: str` (min 10) [FACT: hypothesis.py:165], `num_hypotheses: int` (1-10, default 3) [FACT: hypothesis.py:175], `max_iterations: int` (1-5, default 1) [FACT: hypothesis.py:182], `require_novelty_check: bool` (default True) [FACT: hypothesis.py:183], `min_novelty_score: float` (default 0.5) [FACT: hypothesis.py:184] | Pydantic default serialization | hypothesis_generator input | `min_novelty_score=0.5` differs from `NoveltyReport.novelty_threshold_used=0.75` [FACT: hypothesis.py:259] and `Hypothesis.is_novel(threshold=0.5)` [FACT: hypothesis.py:150] |
| `HypothesisGenerationResponse` | `kosmos/models/hypothesis.py:199-237` | Pydantic BaseModel | `hypotheses: List[Hypothesis]`, `model_used` (defaults to `_DEFAULT_CLAUDE_SONNET_MODEL` from config) [FACT: hypothesis.py:212] | Pydantic default serialization | hypothesis_generator output -> research_director | `get_best_hypothesis()` silently returns first hypothesis as fallback when none have priority_score [FACT: hypothesis.py:229]. `model_used` coupled to config constant [FACT: hypothesis.py:13]. |
| `NoveltyReport` | `kosmos/models/hypothesis.py:240-262` | Pydantic BaseModel | `novelty_score` (0-1), `max_similarity`, `prior_art_detected`, `is_novel`, `novelty_threshold_used=0.75` [FACT: hypothesis.py:259] | Pydantic default serialization | novelty_checker output | Threshold default 0.75 differs from request default 0.5 and method default 0.5 [FACT: hypothesis module findings] |
| `TestabilityReport` | `kosmos/models/hypothesis.py:265-296` | Pydantic BaseModel | `testability_score` (0-1), `is_testable`, `testability_threshold_used=0.3` [FACT: hypothesis.py:277], `primary_experiment_type` | Pydantic default serialization | testability checker output | Threshold 0.3 is surprisingly low [FACT: hypothesis module findings] |
| `PrioritizedHypothesis` | `kosmos/models/hypothesis.py:299-331` | Pydantic BaseModel | Scoring weights: novelty=0.30, feasibility=0.25, impact=0.25, testability=0.20 [FACT: hypothesis.py:315-319] | Pydantic default serialization | prioritizer output | `update_hypothesis_priority()` mutates the embedded `hypothesis.priority_score` and `hypothesis.updated_at` [FACT: hypothesis.py:328-331] -- side effect on nested object |
| `ExperimentProtocol` | `kosmos/models/experiment.py:329-575` | Pydantic BaseModel | `id: Optional[str]` [FACT: experiment.py:351], `hypothesis_id: str` [FACT: experiment.py:353], `experiment_type: ExperimentType` [FACT: experiment.py:355], `steps: List[ProtocolStep]` (min 1) [FACT: experiment.py:363], `variables: Dict[str, Variable]` [FACT: experiment.py:364], `statistical_tests: List[StatisticalTestSpec]` [FACT: experiment.py:368], `sample_size: Optional[int]` (1-100k) [FACT: experiment.py:369], `resource_requirements: ResourceRequirements` (required) [FACT: experiment.py:395], `rigor_score: Optional[float]` (0-1) [FACT: experiment.py:399], `random_seed: Optional[int]` [FACT: experiment.py:402] | Manual `to_dict()` [FACT: experiment.py:471-573]: 100 lines, deep nested serialization. Converts enum `.value` throughout [FACT: experiment.py:477]. `use_enum_values=False` [FACT: experiment.py:575]. | experiment_designer -> code_generator, result_collector, 14 templates; FK target for ExperimentResult | Step numbering must be contiguous 1..N [FACT: experiment.py:435]. `ensure_title` replaces bad titles silently [FACT: experiment.py:181]. Duplicated sample_size validators [FACT: experiment module findings]. 6-level nesting. `to_dict()` and `model_dump()` produce different output. |
| `Variable` | `kosmos/models/experiment.py:42-78` | Pydantic BaseModel | `name: str` [FACT: experiment.py:57], `type: VariableType` [FACT: experiment.py:58], `description: str` (min 10) [FACT: experiment.py:59], `values: Optional[List]` [FACT: experiment.py:64], `unit: Optional[str]` [FACT: experiment.py:67] | Nested in ExperimentProtocol.to_dict() | code_generator reads `v.type.value == 'independent'` [FACT: code_generator.py:74] | String comparison on enum values is brittle |
| `VariableType` | `kosmos/models/experiment.py:21-26` | str Enum | `INDEPENDENT` [FACT: experiment.py:22], `DEPENDENT` [FACT: experiment.py:23], `CONTROL` [FACT: experiment.py:24], `CONFOUNDING` [FACT: experiment.py:25] | `.value` serialization | code_generator filters by string value | Changing string values breaks code_generator's string comparison [FACT: code_generator.py:74] |
| `ControlGroup` | `kosmos/models/experiment.py:81-135` | Pydantic BaseModel | `name`, `description` (min 5) [FACT: experiment.py:96-97], `variables: Dict` [FACT: experiment.py:99], `rationale` (min 10) [FACT: experiment.py:102], `sample_size: Optional[int]` (ge=1) [FACT: experiment.py:104] | Nested in ExperimentProtocol.to_dict() | Nested within ExperimentProtocol | `coerce_sample_size` converts string to int [FACT: experiment.py:113], clamps to 100k [FACT: experiment.py:120-124] |
| `ProtocolStep` | `kosmos/models/experiment.py:138-191` | Pydantic BaseModel | `step_number: int` (ge=1) [FACT: experiment.py:154], `title` (min 3) [FACT: experiment.py:155], `description` (min 10) [FACT: experiment.py:156], `action: str` [FACT: experiment.py:159], `code_template: Optional[str]` [FACT: experiment.py:173], `library_imports: List[str]` [FACT: experiment.py:174] | Nested in ExperimentProtocol.to_dict() | code_generator iterates reading `step.action`, `step.code_template`, `step.library_imports` | `ensure_title` silently replaces empty titles with "Untitled Step" [FACT: experiment.py:181] |
| `ResourceRequirements` | `kosmos/models/experiment.py:193-232` | Pydantic BaseModel | `compute_hours` (ge=0), `memory_gb` (ge=0), `gpu_required` (bool), `estimated_cost_usd` (ge=0) [FACT: experiment.py:209-216], `required_libraries: List[str]` [FACT: experiment.py:227], `can_parallelize: bool` [FACT: experiment.py:231] | Nested in ExperimentProtocol.to_dict() | Cost/feasibility checks in experiment_designer | No cross-field validation (e.g., `gpu_memory_gb` not required when `gpu_required=True`) [FACT: experiment module findings] |
| `StatisticalTestSpec` | `kosmos/models/experiment.py:235-299` | Pydantic BaseModel | `test_type: StatisticalTest` [FACT: experiment.py:251], `null_hypothesis: str` [FACT: experiment.py:255], `alpha: float` (default 0.05) [FACT: experiment.py:257], `correction_method: Optional[str]` [FACT: experiment.py:276] | Nested in ExperimentProtocol.to_dict() | code_generator checks `test.test_type.value` with string matching [FACT: code_generator.py:65-66] | `coerce_groups` splits comma strings [FACT: experiment.py:266-273]. `parse_effect_size` uses regex extraction [FACT: experiment.py:293]. |
| `StatisticalTest` | `kosmos/models/experiment.py:29-39` | str Enum | `T_TEST` [FACT: experiment.py:31], `ANOVA` [FACT: experiment.py:32], `CHI_SQUARE` [FACT: experiment.py:33], `CORRELATION` [FACT: experiment.py:34], `REGRESSION` [FACT: experiment.py:35], `MANN_WHITNEY` [FACT: experiment.py:36], `KRUSKAL_WALLIS` [FACT: experiment.py:37], `WILCOXON` [FACT: experiment.py:38], `CUSTOM` [FACT: experiment.py:39] | `.value` serialization | Consumed by code_generator via string matching | code_generator uses `'t_test' in test_type_str.lower()` [FACT: code_generator.py:65-66] -- brittle |
| `ExperimentDesignRequest` | `kosmos/models/experiment.py:578-613` | Pydantic BaseModel | `hypothesis_id: str` [FACT: experiment.py:592], `preferred_experiment_type` [FACT: experiment.py:595], `max_cost_usd` [FACT: experiment.py:599], `require_control_group: bool` (default True) [FACT: experiment.py:604], `min_rigor_score: float` (default 0.6) [FACT: experiment.py:606] | Pydantic default serialization | experiment_designer input | N/A |
| `ExperimentDesignResponse` | `kosmos/models/experiment.py:616-654` | Pydantic BaseModel | `protocol: ExperimentProtocol`, `design_time_seconds: float`, `model_used` (defaults to config constant) [FACT: experiment.py:627], `validation_passed: bool` [FACT: experiment.py:631], `rigor_score` (0-1) [FACT: experiment.py:636], `feasibility_assessment: str` [FACT: experiment.py:642] | Pydantic default serialization | experiment_designer output -> research_director | `is_feasible()` checks cost AND duration AND `validation_passed` [FACT: experiment.py:648-654] |
| `ValidationReport` | `kosmos/models/experiment.py:657-699` | Pydantic BaseModel | `rigor_score` (0-1) [FACT: experiment.py:664], `checks_passed/failed/warnings: int` [FACT: experiment.py:668-670], `has_control_group: bool` [FACT: experiment.py:673], `sample_size_adequate: bool` [FACT: experiment.py:676], `validation_passed: bool` [FACT: experiment.py:693], `severity_level: str` [FACT: experiment.py:694] | Pydantic default serialization | Internal to experiment validation | Uses `datetime.utcnow` factory [FACT: experiment.py:699] |
| `ExperimentResult` | `kosmos/models/result.py:127-278` | Pydantic BaseModel | `experiment_id: str`, `protocol_id: str`, `status: ResultStatus` [FACT: result.py:141], `supports_hypothesis: Optional[bool]` [FACT: result.py:174-177], `primary_p_value: Optional[float]` [FACT: result.py:170], `primary_effect_size: Optional[float]` [FACT: result.py:171], `statistical_tests: List[StatisticalTestResult]` [FACT: result.py:160-163], `metadata: ExecutionMetadata` (required) [FACT: result.py:180], `variable_results: List[VariableResult]` [FACT: result.py:154-157], `raw_data: Dict` [FACT: result.py:144-147], `version: int` (default 1) [FACT: result.py:183] | `to_dict()` delegates to `model_to_dict(self, mode='json', exclude_none=True)` via compat [FACT: result.py:241-243]. `to_json()` uses `model_dump_json(indent=2, exclude_none=True)` [FACT: result.py:245-247]. `from_dict()` / `from_json()` for round-trip [FACT: result.py:249-257]. | result_collector -> research_director, data_analyst, summarizer, convergence, feedback, visualization; DB serialization boundary | `supports_hypothesis` is tri-state: True/False/None [FACT: result.py:174-177]. `to_dict()` and `to_json()` may differ on edge cases (datetime formatting) [FACT: result module findings]. `export_markdown()` crashes on None stats [FACT: result module findings]. `statistical_tests` requires unique `test_name` values [FACT: result.py:212-213]. No cascading relationship to ExperimentProtocol [FACT: result module findings]. |
| `ResultStatus` | `kosmos/models/result.py:16-22` | str Enum | `SUCCESS` [FACT: result.py:17], `FAILED` [FACT: result.py:18], `PARTIAL` [FACT: result.py:19], `TIMEOUT` [FACT: result.py:20], `ERROR` [FACT: result.py:21] | `.value` serialization | feedback uses for pattern classification | `FAILED` vs `ERROR` distinction: FAILED = ran but no useful results; ERROR = system/runtime error prevented execution [FACT: result module findings] |
| `ExecutionMetadata` | `kosmos/models/result.py:25-58` | Pydantic BaseModel | `start_time`, `end_time: datetime` (required), `duration_seconds: float` (ge=0) [FACT: result.py:28-30], `python_version`, `platform: str` (required) [FACT: result.py:33-34], `experiment_id`, `protocol_id: str` (required) [FACT: result.py:49-50], `hypothesis_id: Optional[str]` [FACT: result.py:51] | Nested in ExperimentResult.to_dict() | result_collector constructs; export_markdown reads `.duration_seconds`, `.python_version` [FACT: result.py:371-372] | All numeric fields have `ge=0` constraints. Stores `sandbox_used`, `timeout_occurred` bools [FACT: result.py:54-55]. |
| `StatisticalTestResult` | `kosmos/models/result.py:61-103` | Pydantic BaseModel | `test_type`, `test_name: str` [FACT: result.py:64-65], `statistic: float` [FACT: result.py:68], `p_value: float` (0-1) [FACT: result.py:69], `effect_size: Optional[float]` [FACT: result.py:72], `significant_0_05`, `significant_0_01`, `significant_0_001: bool` (all required, no defaults) [FACT: result.py:83-85], `significance_label: str` [FACT: result.py:87], `is_primary: bool` (default False) [FACT: result.py:89] | Nested in ExperimentResult | visualization, export_markdown | All three significance booleans required with no defaults -- constructors must compute all three [FACT: result.py:83-85] |
| `VariableResult` | `kosmos/models/result.py:105-124` | Pydantic BaseModel | `variable_name`, `variable_type: str`, optional stats (mean, median, std, min, max), `values: List[Union[float, int, str]]` [FACT: result.py:119], `n_samples`, `n_missing` [FACT: result.py:122-124] | Nested in ExperimentResult | summarizer, export_markdown | `values` stores raw data as list -- no size limit, can be very memory-intensive [FACT: result module findings] |
| `ResultExport` | `kosmos/models/result.py:280-377` | Pydantic BaseModel | `result: ExperimentResult`, `format: str` [FACT: result.py:283-284] | `export_json()`, `export_csv()` (lazy pandas import [FACT: result.py:292]), `export_markdown()` [FACT: result.py:312-377] | CLI report generation | `export_csv()` requires pandas at runtime [FACT: result module findings]. `export_markdown()` crashes on None variable stats due to `:.2f` formatting [FACT: result module findings]. |

### Configuration Models

| Model | File | Type | Key Fields | Serialization | Cross-Boundary Flow | Gotcha |
|-------|------|------|-----------|---------------|--------------------|---------| 
| `KosmosConfig` | `kosmos/config.py:922-1160` | Pydantic BaseSettings | 16 sections + `llm_provider` enum [FACT: config.py:953-976]: `claude`, `anthropic`, `openai`, `litellm`, `local_model`, `research`, `database`, `redis`, `logging`, `literature`, `vector_db`, `neo4j`, `safety`, `performance`, `monitoring`, `development`, `world_model` | `to_dict()` delegates to `model_to_dict()` [FACT: config.py:1103-1133] | Singleton via `get_config()` [FACT: config.py:1140-1154] consumed by 53 importers | Optional provider configs can be `None` [FACT: config.py:896-919]. `validate_provider_config` model validator enforces API key presence [FACT: config.py:1024-1043]. Creates directories on first call [FACT: config.py:1067-1076]. |
| `ClaudeConfig` | `kosmos/config.py:29-84` | Pydantic BaseSettings | `api_key`, `model`, `max_tokens`, `temperature`, `enable_cache`, `is_cli_mode` property [FACT: config.py:37-82] | Part of KosmosConfig | llm.py reads `config.claude.api_key`, `config.claude.model` | `is_cli_mode` returns True when API key is all 9s [FACT: config.py:80-82]. Aliased as `AnthropicConfig` [FACT: config.py:88]. |
| `ResearchConfig` | `kosmos/config.py:200-247` | Pydantic BaseSettings | `max_iterations` [FACT: config.py:203-208], `enabled_domains: List[str]` [FACT: config.py:210-214], `budget_usd` [FACT: config.py:232-237], `max_runtime_hours` (upper bound 24.0) [FACT: config.py:241] | Part of KosmosConfig | research_director reads `max_iterations`, `enabled_domains` | `enabled_domains` uses `parse_comma_separated` BeforeValidator [FACT: config.py:210]. `max_runtime_hours` documentation says 12 hours but validation allows 24 [FACT: config module findings]. |
| `DatabaseConfig` | `kosmos/config.py:250-302` | Pydantic BaseSettings | `url`, `echo: bool`, `normalized_url` (computed property) [FACT: config.py:253-300] | Part of KosmosConfig | db/__init__.py reads `config.database.normalized_url` | `normalized_url` silently converts relative SQLite paths to absolute [FACT: config module findings]. Raw `url` still holds relative path. |
| `LoggingConfig` | `kosmos/config.py:365-432` | Pydantic BaseSettings | `level`, `format`, `debug_mode`, `debug_level`, `log_agent_messages`, `log_workflow_transitions`, `stage_tracking_enabled` [FACT: config.py:368-418] | Part of KosmosConfig | logging.py, workflow.py, base.py read various fields | 12 fields total [FACT: xray structure.classes LoggingConfig field_count=12] |
| `SafetyConfig` | `kosmos/config.py:573-683` | Pydantic BaseSettings | `enable_safety_checks`, `enable_sandboxing`, `approval_mode` + 16 more fields [FACT: config.py:573-683] | Part of KosmosConfig | executor, guardrails | 19 fields [FACT: xray structure.classes SafetyConfig field_count=19] -- most complex config section |
| `LiteLLMConfig` | `kosmos/config.py:143-197` | Pydantic BaseSettings | `model`, `api_key`, `api_base` + 4 more [FACT: config.py:143-197] | Part of KosmosConfig | provider factory | `sync_litellm_env_vars` is a manual workaround [FACT: config.py:986-1022]. `.env` path hardcoded [FACT: config.py:194]. |

### Agent and Orchestration Models

| Model | File | Type | Key Fields | Serialization | Cross-Boundary Flow | Gotcha |
|-------|------|------|-----------|---------------|--------------------|---------| 
| `AgentMessage` | `kosmos/agents/base.py:45-84` | Pydantic BaseModel | `id: str` (auto UUID) [FACT: base.py:60], `type: MessageType` [FACT: base.py:61], `from_agent: str` [FACT: base.py:62], `to_agent: str` [FACT: base.py:63], `content: Dict[str, Any]` [FACT: base.py:64], `correlation_id: Optional[str]` [FACT: base.py:65], `timestamp: datetime` [FACT: base.py:66], `metadata: Dict` [FACT: base.py:67] | `to_dict()` [FACT: base.py:69-80] and `to_json()` [FACT: base.py:82-84] | Inter-agent communication via send_message/receive_message; content is untyped Dict -- no schema validation on message payloads | `content` is `Dict[str, Any]` with no schema validation [FACT: base.py:64]. hypothesis_generator and experiment_designer override `execute()` to accept/return `AgentMessage` instead of `Dict` [FACT: hypothesis_generator.py:91, experiment_designer.py:109] -- latent interface mismatch. |
| `MessageType` | `kosmos/agents/base.py:37-43` | str Enum | `REQUEST` [FACT: base.py:39], `RESPONSE` [FACT: base.py:40], `NOTIFICATION` [FACT: base.py:41], `ERROR` [FACT: base.py:42] | `.value` serialization | Message routing in registry | Error responses auto-sent when `process_message()` raises on REQUEST messages [FACT: base.py:361-366] |
| `AgentStatus` | `kosmos/agents/base.py:26-34` | str Enum | `CREATED` [FACT: base.py:27], `STARTING` [FACT: base.py:28], `RUNNING` [FACT: base.py:29], `IDLE` [FACT: base.py:30], `WORKING` [FACT: base.py:31], `PAUSED` [FACT: base.py:32], `STOPPED` [FACT: base.py:33], `ERROR` [FACT: base.py:34] | `.value` serialization | Health monitoring via `is_healthy()` [FACT: base.py:211-217] and `get_status()` [FACT: base.py:219-240] | `is_healthy()` only returns True for RUNNING, IDLE, WORKING [FACT: base.py:217]. Forgetting status transition causes health check failure. |
| `AgentState` | `kosmos/agents/base.py:87-94` | Pydantic BaseModel | `agent_id`, `agent_type`, `status`, `data: Dict`, `created_at`, `updated_at` [FACT: base.py:89-94] | Pydantic default serialization | `get_state()` / `restore_state()` for persistence | Using `self.state_data[key] = value` directly skips `updated_at` update [FACT: base.py:475]. Must use `save_state_data()`. |
| `PlanReview` | `kosmos/orchestration/plan_reviewer.py:32-52` | dataclass | `approved: bool` [FACT: plan_reviewer.py:34], `scores: Dict[str, float]` [FACT: plan_reviewer.py:35], `average_score: float` [FACT: plan_reviewer.py:36], `min_score: float` [FACT: plan_reviewer.py:37], `feedback: str` [FACT: plan_reviewer.py:38], `required_changes: List[str]` [FACT: plan_reviewer.py:39], `suggestions: List[str]` [FACT: plan_reviewer.py:40] | `to_dict()` [FACT: plan_reviewer.py:42-52] | plan_reviewer -> research_loop; `approved` gates execution | Not Pydantic -- plain dataclass with manual `to_dict()`. `approved` determined by `avg >= min_average_score AND min >= min_dimension_score AND structural_ok` [FACT: plan_reviewer.py:144-148]. |

### Workflow State Models

| Model | File | Type | Key Fields | Serialization | Cross-Boundary Flow | Gotcha |
|-------|------|------|-----------|---------------|--------------------|---------| 
| `WorkflowState` | `kosmos/core/workflow.py:18-29` | str Enum | 9 states: `INITIALIZING`, `GENERATING_HYPOTHESES`, `DESIGNING_EXPERIMENTS`, `EXECUTING`, `ANALYZING`, `REFINING`, `CONVERGED`, `PAUSED`, `ERROR` [FACT: workflow.py:19-29] | `.value` serialization | research_director, convergence, logging, DB records via ResearchSession.status | Removing any value breaks the transition table. CONVERGED is nearly terminal (only exit: GENERATING_HYPOTHESES) [FACT: workflow.py:211-213]. |
| `NextAction` | `kosmos/core/workflow.py:32-42` | str Enum | 8 actions: `GENERATE_HYPOTHESIS`, `DESIGN_EXPERIMENT`, `EXECUTE_EXPERIMENT`, `ANALYZE_RESULT`, `REFINE_HYPOTHESIS`, `CONVERGE`, `PAUSE`, `ERROR_RECOVERY` [FACT: workflow.py:33-42] | `.value` serialization | research_director decision routing | N/A |
| `WorkflowTransition` | `kosmos/core/workflow.py:45-54` | Pydantic BaseModel | `from_state`, `to_state`, `action: str`, `timestamp` (default utcnow), `metadata: Dict` [FACT: workflow.py:46-54] | Uses `ConfigDict(use_enum_values=True)` [FACT: workflow.py:48] -- enums stored as strings | Internal to ResearchWorkflow.transition_history | Opposite enum serialization strategy from Hypothesis (`use_enum_values=False`) [FACT: hypothesis.py:156]. All timestamps use `datetime.utcnow()` (deprecated) [FACT: workflow.py:53]. |
| `ResearchPlan` | `kosmos/core/workflow.py:57-164` | Pydantic BaseModel | `research_question`, `domain`, `current_state` (default INITIALIZING) [FACT: workflow.py:63-65]; Hypothesis tracking: `hypothesis_pool`, `tested_hypotheses`, `supported_hypotheses`, `rejected_hypotheses` (all List[str]) [FACT: workflow.py:68-71]; Experiment tracking: `experiment_queue`, `completed_experiments` [FACT: workflow.py:74-75]; `results: List[str]` [FACT: workflow.py:78]; `iteration_count`, `max_iterations` [FACT: workflow.py:81-82]; `has_converged`, `convergence_reason` [FACT: workflow.py:85-86] | No explicit `to_dict()` -- uses Pydantic default. 20 fields [FACT: xray structure.classes ResearchPlan field_count=20]. | research_director primary state container. All mutation methods update `updated_at` via `update_timestamp()` [FACT: workflow.py:96-98]. | `mark_supported()` implicitly calls `mark_tested()` [FACT: workflow.py:116]. All mutations deduplicate [FACT: workflow.py:102]. No persistence -- memory only [FACT: workflow module findings]. |

### LLM Provider Contracts

| Model | File | Type | Key Fields | Serialization | Cross-Boundary Flow | Gotcha |
|-------|------|------|-----------|---------------|--------------------|---------| 
| `LLMResponse` | `kosmos/core/providers/base.py:57-154` | dataclass | `content: str` [FACT: base.py:60], `usage: UsageStats` [FACT: base.py:61], `model: str` [FACT: base.py:62], `finish_reason: Optional[str]`, `raw_response`, `metadata` [FACT: base.py:63-67] | 25 string-delegation methods [FACT: base.py:80-154] (`__str__`, `strip`, `split`, `find`, `replace`, etc.) make it quack like `str` | Every LLM call returns this. Callers use string methods directly: `response.strip().lower()` [FACT: hypothesis_generator.py:282], `response.find("```python")` [FACT: code_generator.py:912], `response.split('\n')` [FACT: summarizer.py:386] | `isinstance(response, str)` returns False [FACT: providers_base module findings]. String methods return `str`, not `LLMResponse` -- metadata lost after any string operation [FACT: providers_base module findings]. `__iter__` yields single characters, not lines [FACT: providers_base module findings]. |
| `UsageStats` | `kosmos/core/providers/base.py:34-54` | dataclass | `input_tokens`, `output_tokens`, `total_tokens: int` [FACT: base.py:38-40], `cost_usd: Optional[float]`, `model`, `provider`, `timestamp` [FACT: base.py:43-47] | Plain dataclass | Accumulated in LLMProvider._update_usage_stats() | `total_tokens` is NOT auto-computed [FACT: providers_base module findings]. Some async paths leave it at 0 [FACT: anthropic.py:422-426]. Counters are not thread-safe [FACT: providers_base module findings]. |
| `Message` | `kosmos/core/providers/base.py:17-31` | dataclass | `role: str`, `content: str`, `name: Optional[str]`, `metadata: Optional[Dict]` [FACT: base.py:18-25] | Plain dataclass | Input to `generate_with_messages()` | `role` is bare `str`, not enum [FACT: providers_base module findings]. Providers handle system messages differently (Anthropic extracts, OpenAI passes through). |
| `ProviderAPIError` | `kosmos/core/providers/base.py:417-484` | Exception | `provider`, `message`, `status_code`, `raw_error`, `recoverable` (default True) [FACT: base.py:429-436] | N/A | Consumed by async_llm CircuitBreaker and delegation error recovery | `is_recoverable()` has ordering bias -- recoverable patterns checked first, so ambiguous messages default to retry [FACT: providers_base module findings]. 4xx (except 429) non-recoverable [FACT: base.py:459-460]. |

### Convergence and Feedback Models

| Model | File | Type | Key Fields | Serialization | Cross-Boundary Flow | Gotcha |
|-------|------|------|-----------|---------------|--------------------|---------| 
| `ConvergenceMetrics` | `kosmos/core/convergence.py` | Pydantic BaseModel | 18 fields [FACT: xray structure.classes ConvergenceMetrics field_count=18] | Pydantic default | convergence -> research_director | Complex model with many optional metrics |
| `StoppingDecision` | `kosmos/core/convergence.py` | Pydantic BaseModel | 6 fields [FACT: xray structure.classes StoppingDecision field_count=6] | Pydantic default | convergence -> research_director | N/A |
| `ConvergenceReport` | `kosmos/core/convergence.py` | Pydantic BaseModel | 17 fields [FACT: xray structure.classes ConvergenceReport field_count=17] | Pydantic default | convergence -> research_director, summarizer | N/A |
| `FeedbackSignal` | `kosmos/core/feedback.py` | Pydantic BaseModel | 6 fields [FACT: xray structure.classes FeedbackSignal field_count=6] | Pydantic default | feedback -> research_director | N/A |
| `SuccessPattern` | `kosmos/core/feedback.py` | Pydantic BaseModel | 11 fields [FACT: xray structure.classes SuccessPattern field_count=11] | Pydantic default | feedback loop analysis | N/A |
| `FailurePattern` | `kosmos/core/feedback.py` | Pydantic BaseModel | 9 fields [FACT: xray structure.classes FailurePattern field_count=9] | Pydantic default | feedback loop analysis | N/A |
| `Memory` | `kosmos/core/memory.py` | Pydantic BaseModel | 9 fields [FACT: xray structure.classes Memory field_count=9] | Pydantic default | memory system | N/A |

### Literature and Knowledge Models

| Model | File | Type | Key Fields | Serialization | Cross-Boundary Flow | Gotcha |
|-------|------|------|-----------|---------------|--------------------|---------| 
| `PaperMetadata` | `kosmos/literature/base_client.py:36-122` | dataclass (NOT Pydantic) | `id: str` (source-specific) [FACT: base_client.py:44], `source: PaperSource` [FACT: base_client.py:45], `title`, `abstract` [FACT: base_client.py:51-52], `authors: List[Author]` [FACT: base_client.py:53], `doi`, `arxiv_id`, `pubmed_id` [FACT: base_client.py:46-48], `citation_count` [FACT: base_client.py:66], `fields`, `keywords` [FACT: base_client.py:71-72], `raw_data` [FACT: base_client.py:78] | Manual `to_dict()` [FACT: base_client.py:99-122]: converts datetimes, authors to dicts. `raw_data` excluded from serialization [FACT: base_client module findings]. | 18 importers across literature clients, knowledge modules, agents, world model [FACT: base_client module findings] | NOT Pydantic -- no validation, no `model_dump()` [FACT: base_client module findings]. `authors` defaults to `None` before `__post_init__` runs [FACT: base_client.py:53]. `primary_identifier` property prioritizes DOI > arXiv > PubMed > source ID [FACT: base_client.py:91-92]. `raw_data` lost on serialization. |
| `PaperSource` | `kosmos/literature/base_client.py:17-24` | str Enum | `ARXIV`, `SEMANTIC_SCHOLAR`, `PUBMED`, `UNKNOWN`, `MANUAL` [FACT: base_client.py:19-24] | `.value` serialization | All literature clients, world_model/simple.py | `MANUAL` enables hand-added papers, not from API [FACT: base_client module findings] |
| `Author` | `kosmos/literature/base_client.py:27-32` | dataclass | `name: str`, `affiliation: Optional[str]`, `email: Optional[str]`, `author_id: Optional[str]` [FACT: base_client.py:28-32] | Nested in PaperMetadata.to_dict() | Nested within PaperMetadata | No validation at all |

### Safety Models

| Model | File | Type | Key Fields | Serialization | Cross-Boundary Flow | Gotcha |
|-------|------|------|-----------|---------------|--------------------|---------| 
| `Alert` | `kosmos/monitoring/alerts.py:34-61` | dataclass | `name: str`, `severity: AlertSeverity`, `message: str`, `timestamp` (default utcnow), `status: AlertStatus` (default ACTIVE), `details: Dict`, `alert_id: Optional[str]` [FACT: alerts.py:34-49] | `to_dict()` serializes enums via `.value`, timestamp via `.isoformat()` [FACT: alerts.py:51-61] | AlertManager -> notification handlers (log, email, Slack, PagerDuty) | `alert_id` auto-generated as `"{name}_{unix_timestamp}"` [FACT: alerts.py:46-49] -- same-second alerts share ID, causing silent overwrites [FACT: alerts module findings] |
| `AlertSeverity` | `kosmos/monitoring/alerts.py:19-24` | str Enum | `INFO`, `WARNING`, `ERROR`, `CRITICAL` [FACT: alerts.py:19-24] | `.value` serialization | Notification dispatch, PagerDuty filtering | PagerDuty handler silently drops INFO and WARNING [FACT: alerts.py:487-488] |
| `AlertStatus` | `kosmos/monitoring/alerts.py:27-31` | str Enum | `ACTIVE`, `RESOLVED`, `ACKNOWLEDGED` [FACT: alerts.py:27-31] | `.value` serialization | Alert lifecycle management | `ACKNOWLEDGED` does not remove from active_alerts [FACT: alerts.py:257-262] |
| `AlertRule` | `kosmos/monitoring/alerts.py:64-111` | dataclass | `name: str`, `condition: Callable[[], bool]`, `severity: AlertSeverity`, `message_template: str`, `cooldown_seconds: int` (default 300), `last_triggered: Optional[datetime]` [FACT: alerts.py:64-74] | Not serialized | Internal to AlertManager | Broken condition callable silently returns False [FACT: alerts.py:75-93]. 3 of 7 default rules are placeholders [FACT: alerts module findings]. |
| `EmergencyStopStatus` | `kosmos/models/safety` | Pydantic BaseModel | Imported by guardrails [FACT: guardrails.py:21-24] | Via `model_to_dict()` | guardrails -> incident logging | `is_emergency_stop_active()` has a write side effect -- checking status can CAUSE emergency stop [FACT: guardrails module findings] |
| `ExecutionResult` | `kosmos/execution/executor.py:113-159` | Plain class (not Pydantic) | `success: bool`, `return_value`, `stdout`, `stderr`, `error`, `execution_time`, `profile_data`, `data_source` [FACT: executor module findings] | `to_dict()` with defensive profile serialization [FACT: executor.py:153-156] | executor -> result_collector -> ExperimentResult | NOT Pydantic. `data_source` set to `'file'` or `'synthetic'` by executed code [FACT: executor module findings]. Return value extracted from `exec_locals.get('results', exec_locals.get('result'))` [FACT: executor.py:516] -- variable name convention is a silent contract. |

### Serialization Strategy Summary

The codebase uses three distinct serialization strategies across its data models, creating an inconsistency that developers must navigate:

| Strategy | Models Using It | Enum Handling | Gotcha |
|----------|----------------|---------------|--------|
| Manual `to_dict()` with explicit `.value` calls | `Hypothesis` [FACT: hypothesis.py:117-142], `ExperimentProtocol` [FACT: experiment.py:471-573], `AgentMessage` [FACT: base.py:69-80], `PaperMetadata` [FACT: base_client.py:99-122], `Alert` [FACT: alerts.py:51-61] | Enums converted to strings | `model_dump()` returns enum objects; `to_dict()` returns strings. Must use `to_dict()` for JSON-safe output. |
| `model_to_dict(self, mode='json', exclude_none=True)` via compat | `ExperimentResult` [FACT: result.py:241-243], `KosmosConfig` [FACT: config.py:1103-1133], guardrails incidents | Pydantic handles via mode='json' | Different behavior from manual `to_dict()` for edge cases |
| Pydantic default (no explicit `to_dict()`) | `ResearchPlan` [FACT: workflow.py:57-164], convergence/feedback models, config section models | Depends on `use_enum_values` setting | `use_enum_values=True` (WorkflowTransition) vs `False` (Hypothesis, ExperimentProtocol) |

### Cross-Boundary Data Flow Diagram

```
                    Research Question (str)
                           |
                           v
                 HypothesisGenerationRequest
                           |
                           v
                 [hypothesis_generator agent]
                           |
                           v
                 HypothesisGenerationResponse
                  contains: List[Hypothesis]
                           |
              +------------+------------+
              |                         |
              v                         v
      ResearchPlan.hypothesis_pool   ConvergenceMetrics
      (stores hypothesis IDs)        (reads novelty_score,
                                      testability_score)
              |
              v
      ExperimentDesignRequest
      (contains hypothesis_id)
              |
              v
      [experiment_designer agent]
              |
              v
      ExperimentDesignResponse
      contains: ExperimentProtocol
      (nested: ProtocolStep[], Variable{}, StatisticalTestSpec[])
              |
              v
      [code_generator] reads protocol.steps, protocol.variables,
                        protocol.statistical_tests
              |
              v
      ExecutionResult (plain class)
      (return_value, stdout, stderr)
              |
              v
      [result_collector]
              |
              v
      ExperimentResult (Pydantic)
      (contains: StatisticalTestResult[], VariableResult[],
       ExecutionMetadata, supports_hypothesis)
              |
              v
      [convergence, feedback, summarizer, data_analyst]
```

All three pipeline models (`Hypothesis`, `ExperimentProtocol`, `ExperimentResult`) have complementary SQLAlchemy models in `kosmos.db.models` for persistence, with no automatic conversion between the Pydantic and SQLAlchemy representations [FACT: hypothesis module findings: "Dual model system"].
## Error Handling Strategy (see Gotcha #124 for silent except-pass sites)

### Dominant Pattern: Catch-Log-Wrap-Continue

Kosmos employs a **layered catch-log-wrap-continue** strategy as its dominant error handling paradigm. The codebase contains 409 `except Exception` clauses across 106 files (`00_calibration.md`), 68 `except ImportError` sites across 41 files (`00_calibration.md`), and 260 `raise` sites across 78 files (`00_calibration.md`). The prevailing behavior is to catch broadly at the call site, log via `logging.getLogger(__name__)`, then either wrap the exception into a domain-specific error type or degrade gracefully. Exceptions almost never propagate raw to the caller (`00_calibration.md`). A secondary survey confirms 183 `except Exception as e:` occurrences across 50+ files (`exception_taxonomy.md`), making broad exception catching the single most common error handling idiom in the project.

This pattern has a clear rationale: Kosmos is an autonomous research platform where a single failure in one subsystem (e.g., a failed LLM call, a Neo4j timeout, a malformed JSON response) should not crash the entire multi-cycle research loop. The design intentionally prioritizes resilience over strictness. However, this comes at a cost: failures are frequently absorbed rather than surfaced, and the boundary between "handled gracefully" and "silently swallowed" is often blurred.

### Custom Exception Hierarchy

Kosmos defines **9 custom exception classes** plus re-exports of 3 placeholder Anthropic SDK exceptions (`exception_taxonomy.md`). All custom exceptions inherit directly from `Exception`; there is **no shared base class** such as `KosmosError` for the project (`exception_taxonomy.md`). This means there is no way to catch "all Kosmos errors" without catching `Exception`, making it impossible to distinguish application errors from stdlib errors at catch sites (`exception_taxonomy.md`).

The full inheritance tree is:

```
Exception
 +-- ProviderAPIError          (core/providers/base.py:417)
 +-- BudgetExceededError       (core/metrics.py:63)
 +-- CacheError                (core/cache.py:22)
 +-- LiteratureCacheError      (literature/cache.py:19)
 +-- PDFExtractionError        (literature/pdf_extractor.py:22)
 +-- JSONParseError            (core/utils/json_parser.py:21)
 +-- APIError [placeholder]    (core/async_llm.py:23)
 +-- APITimeoutError [placeholder] (core/async_llm.py:26)
 +-- RateLimitError [placeholder]  (core/async_llm.py:29)
```

(`exception_taxonomy.md`)

The `WorldModelError` hierarchy documented in `docs/planning/implementation.md:1571-1605` was never implemented in actual code -- searching `kosmos/` for `class.*WorldModelError` yields 0 hits (`00_calibration.md`). Similarly, no `EntityNotFoundError` exists despite it being specified in the architecture doc (`00_calibration.md`). No enum or constant set of error codes exists anywhere in the codebase -- searching for `class.*ErrorCode|ERROR_CODE` produces 0 hits in `kosmos/` (`00_calibration.md`). Errors are identified purely by exception type and string messages, making programmatic error handling fragile.

### ProviderAPIError: The Most Sophisticated Exception

`ProviderAPIError` at `core/providers/base.py:417-484` is the single custom exception for all LLM provider failures (`00_calibration.md`). It carries five fields: `provider`, `message`, `status_code`, `raw_error`, and `recoverable` (default `True`) (`providers_base.md`, `00_calibration.md`). Its `is_recoverable()` method implements a two-phase classification: first, status codes in the 4xx range (except 429) are classified as non-recoverable (`base.py:459-460`); second, message-pattern matching checks against two tuples -- 9 `recoverable_patterns` (including 'timeout', 'rate_limit', '503') and 10 `non_recoverable_patterns` (including 'json', 'authentication', '401') (`providers_base.md`, `00_calibration.md`).

A notable ordering bias exists: recoverable patterns are checked before non-recoverable patterns (`base.py:476-477`), so an ambiguous message like "invalid rate_limit response" that matches both sets would be classified as recoverable (`providers_base.md`). This is a deliberate design choice: default to retry when uncertain.

The `ProviderAPIError` is consumed by the `CircuitBreaker` in `async_llm.py` and the delegation manager's error recovery in `delegation.py` (`providers_base.md`). The delegation manager at `delegation.py:334-337` checks `isinstance(e, ProviderAPIError) and not e.is_recoverable()` to short-circuit retries on non-recoverable errors -- this is the only exception-driven retry logic in the codebase (`exception_taxonomy.md`).

### Other Custom Exceptions

**BudgetExceededError** at `core/metrics.py:63-85` carries `current_cost`, `limit`, and `usage_percent` attributes and is raised at `metrics.py:801` when the budget threshold is crossed (`exception_taxonomy.md`). Critically, there is no evidence of this exception being caught anywhere in the orchestration layer, meaning a budget overrun could crash the entire research loop rather than gracefully stopping it (`exception_taxonomy.md`).

**JSONParseError** at `core/utils/json_parser.py:21` wraps failed LLM JSON parsing with `original_text` and `attempts` count (`json_parser.py:24-28`). The centralized parser tries 5 strategies -- direct parsing, code-block extraction, regex extraction, single-quote cleanup, and bracket-extraction -- before raising (`json_parser.py:39-46`, `00_calibration.md`). It is caught specifically at `core/llm.py:472` as `except JSONParseError as e:`, one of the few places that catches a custom exception type by name (`exception_taxonomy.md`). In `ClaudeClient.generate_structured()`, exhausted JSON parse retries (max 2) raise `ProviderAPIError(recoverable=False)` (`core_llm.md`).

The three placeholder exceptions (`APIError`, `APITimeoutError`, `RateLimitError`) at `core/async_llm.py:23-31` exist to prevent import failures when the `anthropic` package is not installed. They create dummy classes so `isinstance()` checks compile but never match real Anthropic exceptions (`00_calibration.md`, `exception_taxonomy.md`).

### Raise Site Distribution

Custom exceptions are raised in limited locations. The distribution reveals heavy reliance on stdlib exceptions (`exception_taxonomy.md`):

| Exception | Raise Count | Key Raise Sites |
|-----------|-------------|-----------------|
| `ValueError` | 30+ | config.py, world_model/models.py, hypothesis/prioritizer.py, core/events.py |
| `RuntimeError` | 12+ | orchestration/delegation.py, execution/sandbox.py, oversight/human_review.py |
| `ImportError` | 8+ | core/llm.py, analysis/visualization.py, execution/r_executor.py |
| `ProviderAPIError` | 4 | core/llm.py:481, core/async_llm.py:411, core/providers/factory.py:63/76 |
| `JSONParseError` | 3 | core/utils/json_parser.py:59/70/150 |
| `BudgetExceededError` | 1 | core/metrics.py:801 |
| `NotImplementedError` | 2 | world_model/factory.py:138, domains/neuroscience/neurodegeneration.py:533 |

The delegation manager at `delegation.py:397-552` raises `RuntimeError` in 6 places when required agents are missing, with descriptive messages like "No 'data_analyst' agent was provided to DelegationManager" (`exception_taxonomy.md`). Config validation at `config.py:1024-1037` raises `ValueError` with actionable messages: "Please set OPENAI_API_KEY in your environment or .env file" (`exception_taxonomy.md`).

### Provider Error Wrapping

All 3 LLM providers (Anthropic, OpenAI, LiteLLM) follow an identical error wrapping pattern: catch `Exception`, log via `logger.error()`, then `raise ProviderAPIError(provider_name, message, raw_error=e)` (`00_calibration.md`). Specific wrapping sites include:

- `core/providers/anthropic.py:120-122` wraps init failure as `ProviderAPIError("anthropic", ...)` (`00_calibration.md`)
- `core/providers/anthropic.py:340-342` wraps generation failure identically (`00_calibration.md`)
- `core/providers/openai.py:151-153` wraps init failure as `ProviderAPIError("openai", ...)` (`00_calibration.md`)
- `core/providers/openai.py:281-283` wraps generation failure (`00_calibration.md`)
- `core/providers/litellm_provider.py:305-308` wraps generation failure as `ProviderAPIError("litellm", ...)` (`00_calibration.md`)

However, wrapping is inconsistent: `core/llm.py` and `core/async_llm.py` wrap raw Anthropic exceptions into `ProviderAPIError`, but not all LLM call sites go through these wrappers. Direct Anthropic client usage in `research_director.py:228` (`os.getenv("ANTHROPIC_API_KEY")`) bypasses the wrapping layer (`exception_taxonomy.md`).

### Per-Subsystem Error Handling Behavior

#### LLM Layer: Circuit Breaker and Rate Limiter (see Gotcha #38; see Module Index: core_llm.py)

The `CircuitBreaker` at `core/async_llm.py:51` implements three-state (CLOSED/OPEN/HALF_OPEN) circuit breaking with `failure_threshold=3`, `reset_timeout=60.0s`, and `half_open_max_calls=1` (`async_llm.py:62-67`). After 3 consecutive failures, the breaker opens and blocks all requests for 60 seconds (`async_llm.py:121-125`, `00_calibration.md`).

The `is_recoverable_error()` function at `core/async_llm.py:141` gates retry decisions: `RateLimitError` always retries, `APITimeoutError` always retries, and `APIError` retries unless the message contains `invalid|authentication|unauthorized|forbidden` (`async_llm.py:156-169`, `00_calibration.md`).

The `RateLimiter` at `core/async_llm.py:206` uses a token-bucket algorithm with `max_requests_per_minute=50` and `max_concurrent=5` defaults (`async_llm.py:214-218`), gated by `asyncio.Semaphore` (`async_llm.py:230`, `00_calibration.md`).

The `ClaudeClient.generate()` in `core/llm.py` re-raises exceptions after logging (`llm.py:363-365`, `core_llm.md`). For structured generation, `ClaudeClient.generate_structured()` retries up to `max_retries=2` on `JSONParseError`, bypasses cache on retries (`llm.py:467`), and raises `ProviderAPIError(recoverable=False)` when retries are exhausted (`llm.py:481-486`, `core_llm.md`).

The LLM singleton factory `get_client()` uses double-checked locking via `threading.Lock` (`llm.py:643-649`, `core_llm.md`) for thread-safe initialization, but once the singleton provider is obtained, its usage counters have no protection (`providers_base.md`). The usage counters on `LLMProvider` (`total_input_tokens`, `total_output_tokens`, `total_cost_usd`, `request_count`) are instance-level with no locking (`base.py:190-193`, `providers_base.md`).

#### Executor: Self-Correcting Retry with Code Repair (see Module Index: executor.py; see Critical Path 2)

`RetryStrategy` at `execution/executor.py:667` provides code-aware retry with `max_retries=3` and exponential backoff (`base_delay * 2^(attempt-1)`) (`executor.py:699,732-734`, `00_calibration.md`). It classifies `SyntaxError`, `FileNotFoundError`, and `DataUnavailableError` as non-retryable (`executor.py:724-730`, `00_calibration.md`). On failure, it attempts rule-based code fixes for 10+ error types (missing imports, key errors, type errors) via `modify_code_for_retry()`, with optional LLM-assisted repair on the first 2 attempts (`executor.py:778-786`, `00_calibration.md`).

The `CodeExecutor.execute()` method at `executor.py:237` runs a while loop up to `max_retries` with the `RetryStrategy`, sleeping `delay` seconds between attempts (`executor.py:277-278, 333-335`). If all retries exhaust, it returns `ExecutionResult(success=False, error_type="MaxRetriesExceeded")` (`executor.py:372-376`, `00_calibration.md`). All exceptions are caught -- the method never propagates exceptions to callers (`executor.md`).

The `COMMON_IMPORTS` dict at `executor.py:680-697` maps 16 common variable names to their import statements (e.g., `'pd'` -> `'import pandas as pd'`), used for auto-fixing NameError (`executor.md`). For `FileNotFoundError`, the strategy returns `None` (no fix), marking the error as terminal to prevent infinite retry loops (`executor.md`).

A critical gotcha: most pattern-based fixes wrap code in try/except. If retried again, the already-wrapped code gets wrapped again, producing nested try/except blocks (`executor.md`). Additionally, LLM repair sends user-generated code directly to an LLM prompt via `_repair_with_llm()` -- if the code contains adversarial content, it could manipulate the repair (`executor.md`).

If Docker sandbox is unavailable, `use_sandbox` is silently set to `False` at `executor.py:215-224`, meaning callers who request sandboxing may unknowingly run unsandboxed with no callback or exception (`executor.md`). On Windows, `_exec_with_timeout()` at `executor.py:622` uses `ThreadPoolExecutor` but `future.result(timeout=...)` raises `TimeoutError` without actually killing the exec'd thread -- runaway code continues executing in the background (`executor.md`).

#### Research Director: Error Recovery with Consecutive Error Tracking (see Module Index: research_director.py; see Critical Path 0)

`ResearchDirectorAgent` at `agents/research_director.py:44-46` configures `MAX_CONSECUTIVE_ERRORS=3` with `ERROR_BACKOFF_SECONDS=[2, 4, 8]` (exponential backoff) (`00_calibration.md`, `research_director.md`). The `_handle_error_with_recovery()` method at line 599 increments `_consecutive_errors`, logs with `[ERROR-RECOVERY]` prefix, and either applies backoff + retry (recoverable) or skips to next action (non-recoverable) (`research_director.py:628, 643-645, 664-677`, `00_calibration.md`). When consecutive errors hit 3, the workflow transitions to `WorkflowState.ERROR` (`research_director.py:649-662`, `00_calibration.md`). Successful operations call `_reset_error_streak()` to zero the counter (`research_director.py:686-698`, `00_calibration.md`). This handler is invoked from 10+ message handler methods (`research_director.py:715, 764, 812, 875, 944, 1451, 1514, 1659, 1789, 1971`, `00_calibration.md`).

The research director contains **32 `except Exception as e:` blocks**, the highest count of any single file (`exception_taxonomy.md`). Most follow the pattern: catch, log error, continue to next iteration. An infinite loop prevention guard uses `MAX_ACTIONS_PER_ITERATION = 50` with an `_actions_this_iteration` counter (`research_director.md`).

The director also attempts database initialization in its `__init__` at lines 131-139, swallowing `RuntimeError` if the DB is already initialized. This means every `ResearchDirectorAgent` instance tries to init the DB, creating a race condition if multiple instances are created concurrently (`research_director.md`).

#### Research Loop: Per-Cycle Resilience

`workflow/research_loop.py:235` catches per-cycle `Exception`, logs `f"Cycle {cycle} failed: {e}"`, and continues to the next cycle via `continue` (`research_loop.py:235-245`, `00_calibration.md`). Individual cycle failures do not abort the multi-cycle run.

#### Workflow State Machine: Error as a State

The `ResearchWorkflow` state machine at `core/workflow.py` defines `ERROR` as one of 9 workflow states. The `ALLOWED_TRANSITIONS` dict permits transitions FROM `ERROR` to three recovery states: `INITIALIZING` (restart), `GENERATING_HYPOTHESES` (resume), and `PAUSED` (`core_workflow.md`). When `transition_to()` is called with an invalid target state, it raises `ValueError` with the current state, target state, and allowed transitions (`core_workflow.md`). The config import for transition logging is wrapped in try/except -- if config loading fails, the workflow silently continues without logging (`core_workflow.md`, `00_calibration.md`).

#### Database Layer: Context Manager with Rollback (see Module Index: db.py; see Gotcha #65; see Gotcha #66)

`db/__init__.py:130-137` implements the context manager `get_session()` with correct rollback: `except Exception: session.rollback(); raise` in a `try/finally` with `session.close()`. This is the only place the rollback pattern appears (`00_calibration.md`, `db.md`). However, the rollback at line 134 is not wrapped in its own try/except -- if `session.rollback()` itself fails (e.g., connection lost), the original exception from the yield block is lost (`db.md`).

All CRUD functions in `db/operations.py` call `session.commit()` immediately (`operations.py:113, 220, 363`), meaning operations cannot be batched atomically (`db.md`). The `get_session()` context manager also commits on clean exit (`db/__init__.py:132`), leading to potential double-commits (`db.md`).

If database initialization fails during CLI startup at `cli/main.py:147-165`, the error is logged and displayed to the user, but the CLI does NOT exit -- this is intentional so commands like `--help` and `version` still work (`initialization.md`, `cli_main.md`).

#### World Model: Silent Degradation (see Module Index: world_model_simple.py; see Module Index: knowledge_graph.py; see Gotcha #85; see Gotcha #86)

The world model integration is entirely optional. The `__init__` of `ResearchDirectorAgent` wraps world model initialization in try/except (`research_director.md`). If `get_world_model()` fails, `self.wm = None` and all `_persist_*_to_graph()` methods early-return (`research_director.md`). Knowledge graph persistence failures are silently swallowed (`research_director.md`).

The `Neo4jWorldModel` at `world_model/simple.py` soft-fails on disconnect: at lines 114-116, if Neo4j is not connected, `add_entity()` returns `entity.id` silently without persisting (`world_model_simple.md`). The `KnowledgeGraph` constructor at `knowledge/graph.py` catches connection failures silently -- `self.graph` is set to `None` and `self._connected` to `False`, with no exception raised (`knowledge_graph.md`). Index creation errors are caught per-index and logged at `debug` level, making failures essentially silent in normal operation (`knowledge_graph.md`).

The `get_world_model()` factory at `world_model/factory.py:105-155` silently falls back from Neo4j to `InMemoryWorldModel` when Neo4j is unavailable, meaning callers may not realize they are operating on ephemeral storage (`shared_state.md`).

Additionally, `export_graph()` at lines 670-676 writes an empty export file (`{"entities": [], "relationships": []}`) when Neo4j is disconnected rather than raising an error, which could silently overwrite a good backup with empty data (`world_model_simple.md`). Partial relationship import failures during `import_graph()` are caught per-relationship and logged as warnings, so partial imports can occur silently (`world_model_simple.md`).

#### Orchestration: Mock Fallback on LLM Failure (see Module Index: plan_reviewer.py; see Gotcha #37; see Gotcha #142)

LLM-dependent orchestration components fall back to mock outputs on any exception (`00_calibration.md`):

- `PlanReviewer.review_plan()` catches `Exception` at `orchestration/plan_reviewer.py:160` and returns `self._mock_review(plan, context)` with hardcoded scores (`plan_reviewer.py:259-270`, `00_calibration.md`). When structural requirements pass, the mock always approves because `avg=7.5 >= 7.0` and `min=7.0 >= 5.0` (`plan_reviewer.md`). When they fail, mock always rejects because `avg=6.0 < 7.0` (`plan_reviewer.md`).
- `PlanCreator` calls `_create_mock_plan()` on failure at `orchestration/plan_creator.py:147,196` (`00_calibration.md`).
- `PlanReviewer` also returns mock at `plan_reviewer.py:118` when no LLM client is available (`00_calibration.md`).

A critical consequence: when the LLM is unreachable, `PlanReviewer` returns 5.0/10.0 default scores on parse failure (`plan_reviewer.md`). If `min_average_score` is set to 5.0 or below, all plans auto-approve during an outage with no user-visible signal. The `feedback` field says `"Failed to parse review"` but this is buried in data, not surfaced as an alert (`00_calibration.md`).

The `_parse_review_response()` method at `plan_reviewer.py:229` extracts JSON from LLM text using `find('{')`/`rfind('}')` heuristic. On parse failure or missing `scores` key, it returns a default dictionary with all dimensions set to 5.0 -- borderline scores that almost always reject, creating a silent quality degradation rather than an explicit error signal (`plan_reviewer.md`).

The `DIMENSION_WEIGHTS` at `plan_reviewer.py:69` are dead code -- scoring uses unweighted average despite weights being defined (`plan_reviewer.md`).

#### Config Layer: Validation-Time Errors

`get_config()` raises `pydantic.ValidationError` if required fields are missing or invalid, and raises `ValueError` from model validators if provider API keys are missing (`config.md`). Accessing `config.openai` when `OPENAI_API_KEY` was never set returns `None`, and attribute access on it raises `AttributeError` (`config.md`). The `validate_provider_config` model validator at `config.py:1024-1037` enforces that the selected provider's API key is present with actionable error messages (`config.md`).

#### Safety/Guardrails: Emergency Stop (see Module Index: guardrails.py; see Gotcha #28; see Gotcha #29)

The `SafetyGuardrails.validate_code()` method at `guardrails.py:128-131` raises `RuntimeError` BEFORE any validation occurs if emergency stop is active -- this is a hard block (`guardrails.md`). The `check_emergency_stop()` method raises `RuntimeError` if stop is active, with the error message including the trigger source and reason (`guardrails.md`). Notably, `is_emergency_stop_active()` has a **side effect**: if the flag file exists and the stop is not already active, it TRIGGERS the stop -- meaning merely checking status can cause a state change (`guardrails.md`).

However, `SafetyGuardrails` is not currently wired into the execution pipeline. `CodeExecutor` uses `CodeValidator` directly, and the emergency stop mechanism is not checked during code execution. The guardrails exist but are not enforced in the actual execution path (`guardrails.md`).

#### Alerts: Silent Placeholder Rules

Three of seven default alert rules in `monitoring/alerts.py` are placeholders that never fire: `high_api_failure_rate`, `api_rate_limit_warning`, and `high_experiment_failure_rate` all always return False (`alerts.md`). The `AlertRule.should_trigger()` method catches all exceptions from the condition callable and returns False -- meaning a broken checker is silently treated as "no alert needed" (`alerts.md`). The `AlertManager` has no thread safety (`alerts.md`).

### Graceful Degradation via ImportError Guards

Optional dependencies use a consistent `try/except ImportError` + boolean flag pattern across 68 occurrences in 41 files (`00_calibration.md`):

- `ASYNC_ANTHROPIC_AVAILABLE` flag with placeholder exception classes at `core/async_llm.py:17-31` -- creates dummy `APIError`, `APITimeoutError`, `RateLimitError` classes so `isinstance()` checks compile but never match (`00_calibration.md`)
- `SANDBOX_AVAILABLE` flag at `execution/executor.py:24-29` -- logs warning and falls back to restricted-builtins execution (`00_calibration.md`)
- `HAS_FASTAPI` flag at `api/websocket.py:14-22` -- sets `APIRouter = None` and `WebSocket = None` (`00_calibration.md`)
- `TENACITY_AVAILABLE` flag at `core/async_llm.py:34-44` -- tenacity retry decorator is optional (`00_calibration.md`)
- `R_EXECUTOR_AVAILABLE` flag at `execution/executor.py:32-37` -- R execution silently disabled (`00_calibration.md`)
- `HAS_ANTHROPIC` flag in `core/llm.py` with warning printed to stdout if SDK is not available (`core_llm.md`)

The `core/providers/factory.py:216-217` auto-registers builtin providers at import time, each wrapped in try/except ImportError (`initialization.md`). Similarly, command registration in `cli/main.py:397-419` imports 7 command modules, silently skipping on `ImportError` -- but because they are imported in a single `from ... import` block, one broken module disables all 7 commands (`cli_main.md`).

### JSON Parse Resilience

JSON parsing from LLM responses uses defensive extraction in 4+ locations (`00_calibration.md`):

- `plan_reviewer.py:232-238` extracts JSON via `find('{')` / `rfind('}')` then `json.loads()`, falling back to default scores on `json.JSONDecodeError` (`00_calibration.md`)
- `core/utils/json_parser.py:31-59` provides a centralized `parse_json_response()` with 5 cascading strategies before raising `JSONParseError` (`00_calibration.md`)
- `hypothesis/refiner.py:220-221` uses same `find/rfind` bracket extraction for JSON from Claude responses (`00_calibration.md`)
- `litellm_provider.py:451-457` has its own JSON cleaning that strips markdown code blocks instead of using the shared `parse_json_response` utility (`providers_base.md`)

### Silent Failures and Swallowed Exceptions

This is the most consequential deviation from the dominant strategy. `except Exception: pass` (bare catch with no logging) appears at 8+ locations in production code (`00_calibration.md`), and 38+ silent `except Exception: pass` sites exist in total (`exception_taxonomy.md`).

**Neo4j diagnostic probing**: `world_model/simple.py:926,937,955` -- three nested `except Exception: pass` blocks in `_get_storage_size_mb()` probing APOC, db.stats, and count-estimation in sequence for Neo4j capabilities (`00_calibration.md`, `exception_taxonomy.md`).

**Config-loading swallowing**: Three providers silently swallow config-loading exceptions with `except Exception: pass` -- `core/providers/anthropic.py:416-417`, `core/providers/openai.py:356-357`, and `core/workflow.py:297-298` (`00_calibration.md`). This means LLM call logging and workflow transition logging can be silently disabled with no diagnostic trail.

**Agent message logging**: `agents/base.py:286-287` and `base.py:351` -- `send_message()` silently swallows config-loading failures: `except Exception: pass  # Config not available` (`exception_taxonomy.md`).

**Literature search results**: `literature/unified_search.py:182-183,279-280,317-318` -- three `except Exception: pass/continue` blocks silently skip individual search results that fail to parse, potentially losing data (`exception_taxonomy.md`).

**Anthropic usage tracking**: `core/providers/anthropic.py:177-178,416-417` -- two `except Exception: pass` blocks silently ignore usage tracking failures (`exception_taxonomy.md`).

**Knowledge/vector DB**: `knowledge/vector_db.py:94-95` -- `except Exception: pass` when deleting a ChromaDB collection during reset (`00_calibration.md`).

**Execution provenance**: `execution/provenance.py:36-37` -- `except Exception: pass` (`00_calibration.md`).

**R executor**: `execution/r_executor.py:358-359` -- `except Exception: pass` (`00_calibration.md`).

**Experiment cache**: `get_cached_result()` in `experiment_cache.py` returns `None` on any exception (silently swallowed) and `find_similar()` returns empty list on exception (`experiment_cache.md`). Only `cache_result()` re-raises exceptions (`experiment_cache.md`).

**Event bus**: `core/event_bus.py:151-152` -- subscriber errors are caught individually and logged, preventing one bad subscriber from breaking all event delivery (`exception_taxonomy.md`).

**Inconsistent logging levels**: Some swallowed exceptions use `logger.debug()` (`workflow/research_loop.py:536`, `api/websocket.py:88`), while identical patterns use bare `pass` with no logging at all (`core/workflow.py:297-298`, `world_model/simple.py:926-927`). No consistent policy exists for which level to use (`00_calibration.md`).

### Initialization Failure Strategy: Four-Service Graceful Degradation

Kosmos follows a pattern where four independent services each initialize with try/except and continue on failure during startup (`initialization.md`):

1. **Database**: `cli/main.py:143-165` -- init failure logged, CLI does not exit (`initialization.md`)
2. **World model**: `research_director.py:241-254` -- catches all exceptions and continues without graph persistence (`initialization.md`)
3. **Async LLM client**: `research_director.py:222-239` -- falls back gracefully if module or API key unavailable (`initialization.md`)
4. **Parallel executor**: `research_director.py:207-219` -- falls back to sequential if import fails (`initialization.md`)

This means the system can run in a degraded state without clear visibility into what is missing (`initialization.md`). There is no single "are all required services ready?" check at startup -- the `doctor` command exists but must be run manually (`initialization.md`).

### CLI Entry Point Error Handling

The `cli_entrypoint()` at `cli/main.py:422-436` wraps `app()` in a try/except, catches `KeyboardInterrupt` (exit 130), and in non-debug mode catches all exceptions with a user-friendly error message (`initialization.md`, `cli_main.md`). Debug detection at line 430 uses `"--debug" in sys.argv` -- a raw argv check (`cli_main.md`).

### Deviations from the Dominant Strategy

1. **World model uses `ValueError` not `ProviderAPIError`**: `world_model/models.py` raises `ValueError` for validation (lines 63, 65, 157, 170, 180, 538, 542, 555), and `world_model/simple.py:427,454,533,1022` raises `ValueError("Entity not found: ...")`. No custom `EntityNotFoundError` exists (`00_calibration.md`). The error contract is inconsistent: some methods return `None` on not-found, others raise `ValueError` (`world_model_simple.md`).

2. **Blocking `time.sleep()` in async-adjacent code**: `research_director.py:674` uses synchronous `time.sleep(backoff_seconds)` for error recovery backoff despite the class having async locks and async LLM clients (`research_director.py:193-195`). This blocks the event loop if called from an async context (`00_calibration.md`, `research_director.md`).

3. **`RuntimeError` overuse in delegation**: The delegation manager raises generic `RuntimeError` for 6 different agent-not-found conditions. A custom `AgentNotFoundError` would be more catchable and descriptive (`exception_taxonomy.md`).

4. **Circuit breaker and error recovery are independent**: The `CircuitBreaker` in `async_llm.py` and the `_handle_error_with_recovery` in `research_director.py` both track consecutive failures with threshold=3 but share no state (`async_llm.py:68` vs `research_director.py:45`). A research director could retry while the circuit breaker is open, generating additional failures (`00_calibration.md`).

5. **Executor `cache_result()` re-raises while `get_cached_result()` swallows**: Within the same `ExperimentCache` class, `cache_result()` propagates exceptions to callers while `get_cached_result()` returns `None` on any exception (`experiment_cache.md`).

6. **Pydantic model validators use defensive LLM parsing**: `ControlGroup.coerce_sample_size` converts strings to int and clamps to `_MAX_SAMPLE_SIZE`, returning `None` on parse failure (`experiment.md`). `StatisticalTestSpec.parse_effect_size` extracts floats via regex (`experiment.md`). These defensive validators absorb malformed LLM output rather than failing.

7. **`generate_with_messages()` ignores auto model selection**: It always uses `self.model`, not the complexity-based selection (`core_llm.md`). This is undocumented behavior.

### Uncaught Paths and Coverage Gaps

The exception taxonomy reveals several paths where exceptions either go uncaught or have no handler:

- **`BudgetExceededError` uncaught**: Raised only in `metrics.py`, with no evidence of being caught in the orchestration layer (`exception_taxonomy.md`)
- **`ProviderAPIError` wrapping bypass**: Direct Anthropic client usage in `research_director.py` bypasses the wrapping layer (`exception_taxonomy.md`)
- **`export_markdown()` latent crash**: The `:.2f` format strings in `result.py:350-353` do not handle `None` values, which are valid per the `VariableResult` model (`result.md`)
- **`get_session()` rollback masking**: If `session.rollback()` fails, the original exception is lost (`db.md`)
- **Signal handler thread restriction**: If `SafetyGuardrails` is instantiated from a non-main thread, signal handler registration silently fails (`guardrails.md`)
- **No tests for `ProviderAPIError.is_recoverable()` logic**: The core retry classification is untested (`providers_base.md`)
- **No tests for base types**: `LLMResponse` string-compat methods, `UsageStats` defaults, and `LLMProvider._update_usage_stats` have no unit tests (`providers_base.md`)
## Shared State

### Dominant Pattern: Module-Level Singleton via Lazy Factory

Kosmos uses a **module-level singleton pattern** as its primary shared state mechanism. At least 14 subsystems expose a global mutable singleton via a `get_X()` factory function that lazily creates an instance on first call and stores it in a module-level `_X` variable (`shared_state.md`). Each singleton typically has a corresponding `reset_X()` function for testing, though 7 of the 14 lack reset functions entirely (see Gotcha #126; see Gotcha #127).

### Inventory of Global Singletons

The following table enumerates every module-level singleton discovered across the codebase (`shared_state.md`):

| Singleton | Module | Variable | Accessor | Reset Function |
|-----------|--------|----------|----------|----------------|
| Config | `config.py:1137` | `_config` | `get_config()` | `reset_config()` |
| DB Engine | `db/__init__.py:22` | `_engine` | `init_database()` | via `global _engine` |
| DB Session | `db/__init__.py:22` | `_SessionLocal` | `get_session()` | via `global _SessionLocal` |
| Event Bus | `core/event_bus.py:261` | `_event_bus` | `get_event_bus()` | `reset_event_bus()` |
| Metrics | `monitoring/metrics.py:448` | `_metrics_collector` | `get_metrics_collector()` | (none) |
| Alerts | `monitoring/alerts.py:525` | `_alert_manager` | `get_alert_manager()` | (none) |
| World Model | `world_model/factory.py:52` | `_world_model` | `get_world_model()` | `reset_world_model()` |
| Stage Tracker | `core/stage_tracker.py:244` | `_tracker` | `get_stage_tracker()` | (none visible) |
| Experiment Cache | `core/experiment_cache.py:726` | `_experiment_cache` | (factory) | (none) |
| Claude Cache | `core/claude_cache.py:373` | `_claude_cache` | `get_claude_cache()` | (none) |
| Agent Registry | `agents/registry.py:509` | `_registry` | (factory) | (none) |
| Health Checker | `api/health.py:398` | `_health_checker` | (factory) | (none) |
| Provider Registry | `core/providers/factory.py:16` | `_PROVIDER_REGISTRY` | `register_provider()` | (none) |
| Literature Analyzer | `agents/literature_analyzer.py:1050` | `_literature_analyzer` | (factory) | (none) |
| LLM Client | `core/llm.py:608-609` | `_default_client` | `get_client()` | `get_client(reset=True)` |
| Knowledge Graph | `knowledge/graph.py:1000` | `_knowledge_graph` | `get_knowledge_graph()` | `reset_knowledge_graph()` |

### Singleton Initialization Details

**Config singleton** (`config.py:1150-1154`): `get_config()` uses `global _config` with lazy initialization: `if _config is None or reload: _config = KosmosConfig()`. The `reload` parameter forces re-creation from environment (`shared_state.md`). On first call (or reload), it also calls `create_directories()` which touches the filesystem (`config.md`). Config is first invoked inside `init_from_config()` during the CLI callback, but provider registration happens at `factory.py` import time (earlier) -- if a provider class tried to read config during registration, it would fail (`initialization.md`).

**Database engine** (`db/__init__.py:67`): `init_database()` uses `global _engine, _SessionLocal` to set module-level SQLAlchemy engine and session factory. This is called once at startup and shared across all database operations (`shared_state.md`, `db.md`). For SQLite, it skips pooling and uses `check_same_thread=False` (`db/__init__.py:72-78`). For PostgreSQL, it uses `QueuePool` with `pool_pre_ping=True` for connection health checks (`db/__init__.py:80-89`, `database_storage.md`). The session factory uses `autocommit=False, autoflush=False` (`db/__init__.py:91`, `database_storage.md`), meaning explicit commits are required.

**Event Bus** (`core/event_bus.py:261-274`): `_event_bus: Optional[EventBus] = None` with `get_event_bus()` creating a singleton `EventBus` instance. The `reset_event_bus()` at line 277 clears all subscribers before nullifying (`shared_state.md`).

**World Model** (`world_model/factory.py:105-155`): `get_world_model()` uses `global _world_model` with Neo4j-to-InMemory fallback: if Neo4j is unavailable, it silently falls back to `InMemoryWorldModel`. The `reset_world_model()` at line 181 calls `close()` before clearing (`shared_state.md`). Callers may not realize they are operating on ephemeral storage when the fallback activates (`shared_state.md`).

**Alert Manager** (`monitoring/alerts.py:525-552`): `_alert_manager` singleton has side effects on creation: `get_alert_manager()` registers default handlers (log, email, slack, pagerduty) based on env vars during instantiation (`shared_state.md`). Handler registration happens once at singleton creation -- if environment variables change after first call, the handler set does not update (`alerts.md`). This makes behavior dependent on initialization order.

**LLM Client** (`core/llm.py:608-610`): `_default_client` typed `Optional[Union[ClaudeClient, LLMProvider]]` with `_client_lock = threading.Lock()` (`core_llm.md`). `get_client()` uses double-checked locking (`llm.py:643-649`). Falls back to `AnthropicProvider` with defaults on config failure (`llm.py:663-673`, `core_llm.md`). `get_provider()` type-checks the result and raises `TypeError` if not `LLMProvider` (`llm.py:703-704`, `core_llm.md`). Configuration changes via `get_client(reset=True)` affect all subsequent callers globally (`core_llm.md`).

**Knowledge Graph** (`knowledge/graph.py:999-1038`): `_knowledge_graph` module-level variable at line 1000 holds the singleton. `get_knowledge_graph()` creates it lazily. `reset_knowledge_graph()` sets it to `None` for testing (`knowledge_graph.md`). The constructor may trigger Docker operations -- calling `_ensure_container_running()` shells out to `docker` and `docker-compose` to start a Neo4j container (`knowledge_graph.md`).

**Provider Registry** (`core/providers/factory.py:16`): `_PROVIDER_REGISTRY: Dict[str, type] = {}` starts empty but is mutated via `register_provider()`. It is populated at import time by `_register_builtin_providers()` with 7 names: `anthropic`, `claude` (alias), `openai`, `litellm`, `ollama` (alias), `deepseek` (alias), `lmstudio` (alias) (`shared_state.md`, `providers_base.md`). This is a mutable module-level dict used as a service locator (`shared_state.md`).

**Experiment Cache** (`core/experiment_cache.py:726`): `_experiment_cache` module-level `Optional[ExperimentCache]` initialized to `None`. Parameters to `get_experiment_cache()` are only applied on first call -- subsequent calls with different thresholds return the same instance with the original configuration (`experiment_cache.md`). `reset_experiment_cache()` sets the global to `None` but does **not** close the SQLite connection or clean up the database file (`experiment_cache.md`).

### Missing Reset Functions

Seven of the 14 singletons have no `reset_*()` function: `MetricsCollector`, `AlertManager`, `StageTracker`, `ExperimentCache`, `ClaudeCache`, `AgentRegistry`, and `HealthChecker` (`shared_state.md`). This makes test isolation difficult for those subsystems. Tests that exercise these singletons will carry state across test boundaries unless manually cleaned up.

### Thread Safety of Singletons

Thread safety across the singleton inventory is inconsistent:

**Thread-safe singletons**:
- `EventBus` uses both `asyncio.Lock` and `threading.Lock` for dual sync/async safety. All `subscribe`/`unsubscribe`/`publish` operations acquire `self._sync_lock` (`shared_state.md`, `core/event_bus.py:54-56`).
- `CacheStats` uses `threading.Lock` for thread-safe increment operations on hit/miss/set/eviction counters (`shared_state.md`, `core/cache.py:30-33`).
- `LLM Client` uses `threading.Lock` for double-checked locking during initialization (`core_llm.md`).
- `ExperimentCache` uses `threading.RLock()` (reentrant lock) for all public methods (`experiment_cache.md`).

**Not thread-safe singletons**:
- `get_metrics_collector()` uses `global _metrics_collector` but has **no thread safety** on the lazy initialization check. Two threads could race to create the singleton (`shared_state.md`).
- `AlertManager` uses no locks. Concurrent calls to `evaluate_rules()` and `resolve_alert()` could produce inconsistent state in `active_alerts` (`alerts.md`).
- `KnowledgeGraph` singleton `get_knowledge_graph()` has a race condition on first access (`knowledge_graph.md`).
- Most `get_*()` factory functions are not thread-safe. The pattern `if _X is None: _X = create()` can race under concurrent access. Only `EventBus` and `LLM Client` address this with locks (`shared_state.md`).

**Partial thread safety**:
- `ExperimentCache` acquires `self._lock` for public methods but `_increment_stat()` at line 576 opens its own separate connection and does **not** acquire the lock, so stat updates can race with other operations (`experiment_cache.md`).
- `LLMProvider` usage counters (`total_input_tokens`, `total_output_tokens`, `total_cost_usd`, `request_count`) are instance-level with no locking (`base.py:190-193`, `providers_base.md`). `self.request_count += 1` is not atomic in CPython for all operations. In concurrent async scenarios, counts could be inaccurate (`providers_base.md`).

### Database Session Lifecycle

Kosmos uses SQLAlchemy with a global engine/session factory pair, initialized once at startup (`database_storage.md`). Sessions are acquired per-operation via the `@contextmanager` function `get_session()` that handles commit/rollback/close automatically (`database_storage.md`).

**Session creation**: `get_session()` at `db/__init__.py:108-137` creates a session from `_SessionLocal()`, yields it, commits on success, rolls back on exception, and always closes in `finally` (`database_storage.md`). If `_SessionLocal is None`, it raises `RuntimeError("Database not initialized. Call init_database() first.")` (`database_storage.md`).

**Session-per-operation pattern**: Each database operation opens and closes its own session. In `research_director.py`, `with get_session() as session:` appears 15+ times at lines 400, 451, 491, 842, 1245, 1293, 1551, 1621, 1690, 1825, 1905, 1928, 2020, 2085 (`database_storage.md`). `hypothesis_generator.py` uses three separate `with get_session()` blocks for creating, querying, and updating hypotheses at lines 474, 513, 559 (`database_storage.md`). A single research step may open 3-5 separate sessions, which is safe but potentially wasteful (`database_storage.md`).

**Synchronous-only sessions**: Despite the codebase using async extensively for agents and LLM calls, database sessions are entirely synchronous. `get_session()` returns a sync SQLAlchemy `Session`, not `AsyncSession`. This means DB writes inside async methods block the event loop (`database_storage.md`, `async_boundaries.md`).

**Connection pooling**: For SQLite, the default configuration explicitly skips connection pooling and uses `check_same_thread=False`. For production PostgreSQL, `QueuePool` with `pool_pre_ping=True` is configured with `pool_size=5`, `max_overflow=10`, `pool_timeout=30` (`database_storage.md`, `db.md`). The `pool_size` parameter is silently ignored for SQLite (`db.md`).

**Schema lifecycle**: `Base.metadata.create_all(bind=_engine)` is called every `init_database()` invocation (`db/__init__.py:100`, `database_storage.md`), ensuring tables exist even without migrations. This is idempotent but schema changes still require Alembic (`db.md`). `reset_database()` drops ALL tables and recreates them -- destructive, intended only for testing/evaluation (`database_storage.md`).

**Slow query monitoring**: `log_slow_queries()` at `db/operations.py:51-80` attaches SQLAlchemy event listeners to measure query duration, logging any query exceeding the threshold (default 100ms) (`database_storage.md`). Batch operations (`executemany`) are not monitored (`db.md`).

### Instance-Level Mutable State

Beyond module-level singletons, several classes maintain significant internal mutable state (`shared_state.md`):

**ArtifactStateManager** (`world_model/artifacts.py:186-187`): Maintains `self._findings_cache: Dict[str, Finding]` and `self._hypotheses_cache: Dict[str, Hypothesis]` as in-memory caches that shadow database content (`shared_state.md`).

**PackageResolver** (`execution/package_resolver.py:207-208`): Keeps `self._installed_cache: Set[str]` and `self._failed_cache: Set[str]` tracking which packages have been installed or failed, with a `clear_cache()` method at line 413 (`shared_state.md`).

**DockerContainerManager** (`execution/docker_manager.py:117`): Holds `self._container_pool: Dict[str, ContainerInstance]` as mutable shared state for pre-warmed Docker containers (`shared_state.md`).

**ResearchDirectorAgent** (`agents/research_director.py:141-154`): Maintains `self.agent_registry: Dict[str, str]`, `self.pending_requests: Dict[str, Dict]`, lazy agent slots (`_hypothesis_agent`, `_experiment_designer`, etc.), and `self.strategy_stats` as mutable research coordination state (`shared_state.md`). Sub-agents are initialized to `None` and only instantiated on first use (`research_director.md`).

**ResearchPlan** (`core/workflow.py:57-164`): All workflow progress tracking state -- `hypothesis_pool`, `tested_hypotheses`, `supported_hypotheses`, `rejected_hypotheses`, `experiment_queue`, `completed_experiments`, `results` -- is held as mutable lists on this Pydantic model. No persistence mechanism exists; if the process crashes, all state is lost (`core_workflow.md`).

**ResearchWorkflow** (`core/workflow.py:166-416`): The `transition_history` list grows unboundedly. A long-running autonomous session could accumulate thousands of transitions in memory with no maximum history size (`core_workflow.md`).

**AlertManager** (`monitoring/alerts.py:125-135`): `alert_history: List[Alert]` is trimmed to 1000 entries when it exceeds that count (`alerts.md`). `active_alerts: Dict[str, Alert]` is keyed by `alert_id`, and same-name alerts triggered within one second share an ID, causing silent overwrites (`alerts.md`).

### Module-Level Constants (Effectively Immutable)

Not all module-level state is mutable. Several modules define frozen constants that function as configuration (`shared_state.md`):

- `research_director.py:45-50` -- `MAX_CONSECUTIVE_ERRORS = 3`, `ERROR_BACKOFF_SECONDS = [2, 4, 8]`, `MAX_ACTIONS_PER_ITERATION = 50` are effectively immutable configuration that should arguably be in the config system (`shared_state.md`)
- `executor.py:40` -- `DEFAULT_EXECUTION_TIMEOUT = 300` seconds (`executor.md`)
- `executor.py:43-83` -- `SAFE_BUILTINS` whitelist of ~70 Python builtins for the restricted exec environment (`executor.md`)
- `executor.py:86-94` -- `_ALLOWED_MODULES` restricts imports to scientific/data libraries plus safe stdlib modules (`executor.md`)
- `experiment.py:19` -- `_MAX_SAMPLE_SIZE = 100_000` hard upper bound for sample sizes (`experiment.md`)

### Async/Sync Boundary State

Kosmos uses a consistent "async-first with sync wrappers" pattern throughout its agent subsystem (`async_boundaries.md`). Every core async method has a corresponding `_sync` wrapper that detects whether an event loop is already running and adapts accordingly.

**The canonical boundary-crossing pattern** appears identically in `BaseAgent`, `AgentRegistry`, and `ResearchDirectorAgent`: (1) try `asyncio.get_running_loop()`, (2) if yes, use `asyncio.run_coroutine_threadsafe(coro, loop)` with a timeout, (3) if no (RuntimeError), create a fresh loop via `asyncio.run(coro)` (`async_boundaries.md`). This pattern is copy-pasted across 6+ locations rather than extracted into a utility function (`async_boundaries.md`).

**Timeout inconsistency**: Most sync wrappers use 30s timeout (`base.py:314-327, 369-380, 393-404`), but `execute_sync()` in the ResearchDirector uses 600s (`research_director.py:2915-2920`), reflecting longer research operations. There is no centralized timeout configuration (`async_boundaries.md`).

**Dual message queues**: Every agent maintains both a sync list `self.message_queue: List[AgentMessage]` and an async `self._async_message_queue: asyncio.Queue[AgentMessage]` simultaneously (`base.py:136-137`, `async_boundaries.md`). The `receive_message()` method writes to **both** queues: `self.message_queue.append(message)` and `await self._async_message_queue.put(message)` (`base.py:337-338`, `async_boundaries.md`). If a consumer reads from the wrong queue, messages could be processed twice (`async_boundaries.md`).

**Dual locking systems in ResearchDirector**: Asyncio locks for async code (`self._research_plan_lock`, `self._strategy_stats_lock`, `self._workflow_lock` -- all `asyncio.Lock`) and threading locks for sync contexts (`self._research_plan_lock_sync`, `self._strategy_stats_lock_sync`, `self._workflow_lock_sync`) coexist (`research_director.md`). The async-compatible context manager `_workflow_context()` at lines 377-379 is a no-op that `yield self.workflow` without actual locking, providing no thread safety for async code paths (`research_director.md`).

**CLI entry point as the primary async/sync boundary**: `kosmos/cli/commands/run.py:186` -- `results = asyncio.run(run_with_progress_async(...))` is the main entry point where sync CLI code crosses into the async research pipeline (`async_boundaries.md`).

**Async LLM client state**: The `AsyncClaudeClient` is conditionally initialized only when `enable_concurrent` is True and `ANTHROPIC_API_KEY` is set (`research_director.py:222-239`, `async_boundaries.md`). The `AnthropicProvider` lazily initializes `AsyncAnthropic` via a property, sharing the same API key and `base_url` as the sync client (`anthropic.py:344-359`, `async_boundaries.md`).

**Legacy test patterns**: Test files use `asyncio.get_event_loop().run_until_complete()` -- the older pre-3.10 idiom (`test_artifacts.py:90`, `test_production_executor.py:81,107`, `async_boundaries.md`). This is deprecated in Python 3.12+ (`async_boundaries.md`). The `execute_experiments_batch()` fallback in the research director also uses `asyncio.get_event_loop().run_until_complete()` at line 2171, which will warn or fail in Python 3.12+ (`research_director.md`).

### Initialization Order Dependencies

The initialization sequence creates implicit ordering constraints between singletons (`initialization.md`):

1. **dotenv** (module-level import of `cli/main.py`) -- populates env vars
2. **Provider registry** (module-level via `factory.py` import) -- registers available LLM backends
3. **Logging** (Typer callback) -- configures handlers
4. **Config singleton** (callback, via `init_from_config` -> `get_config`) -- validates env vars
5. **Database** (callback) -- .env setup, migrations, engine creation, table creation
6. **Command modules** (module-level registration, post-callback import)
7. **Research director** (command-level) -- agents, world model, async client
8. **Agent registry** (command-level) -- message routing setup
9. **Async event loop** (command-level) -- `asyncio.run()` for research

The config singleton timing creates a risk: `get_config()` is first called inside `init_from_config()` during the callback, but provider registration happens at `factory.py` import time (earlier). If a provider class tried to read config during registration, it would fail (`initialization.md`). The flat config bridging -- conversion from nested `KosmosConfig` to flat `Dict[str, Any]` for agents at `run.py:148-170` -- is manual and error-prone; new config fields must be explicitly added (`initialization.md`).

Module-level side effects at import time include: `load_dotenv()` at `cli/main.py:19` populating `os.environ` (`cli_main.md`), `install_rich_traceback()` at `cli/main.py:35` installing a global Rich exception handler (`cli_main.md`), and `register_commands()` at `cli/main.py:419` importing command modules (`cli_main.md`). Importing `kosmos.cli.main` in a test mutates `os.environ` from `.env` file (`cli_main.md`).

### Caches and Connection Pools

**Claude Cache** (`core/claude_cache.py:373`): Singleton LLM response cache. Cache key granularity includes `system`, `max_tokens`, `temperature`, `stop_sequences` -- same prompt with different params misses cache (`core_llm.md`). `reset_stats()` zeroes counters but `ClaudeCache` persists (`core_llm.md`).

**Experiment Cache** (`core/experiment_cache.py`): Standalone SQLite-backed cache with SHA256 fingerprinting for exact match and cosine-similarity-based approximate reuse detection (`experiment_cache.md`). Each method creates and closes its own `sqlite3.connect()` -- no connection pooling. `INSERT OR REPLACE` silently overwrites previous entries, and the `total_experiments` stat counter inflates over time because it increments on overwrites (`experiment_cache.md`). The linear similarity scan `find_similar()` loads all embeddings into memory -- O(n) with no vector index (`experiment_cache.md`).

**Database connection pool**: For PostgreSQL, `QueuePool` with `pool_pre_ping=True`, `pool_size=5`, `max_overflow=10`, `pool_timeout=30` (`db.md`). For SQLite, no connection pooling (`db.md`).

**AnthropicProvider response caching**: Has response caching via `ClaudeCache` (`anthropic.py:129-132`, `providers_base.md`). Cache hit returns immediately, incrementing `cache_hits`; miss increments `cache_misses`, calls the API, then stores the response in cache (`core_llm.md`).

**PackageResolver caches**: `self._installed_cache: Set[str]` and `self._failed_cache: Set[str]` track package installation state in memory (`shared_state.md`).

### Hidden Fallback Behaviors in Shared State

Several singletons have hidden fallback behaviors that change their semantics without caller awareness:

- `get_world_model()` silently falls back from Neo4j to `InMemoryWorldModel` (`shared_state.md`). Callers calling `add_entity()` may believe they are persisting data when they are actually writing to ephemeral memory.
- `get_client()` falls back to `AnthropicProvider` with defaults on config failure (`core_llm.md`). The fallback client may have different model/parameter defaults than what was configured.
- `CodeExecutor` silently downgrades from sandbox to restricted exec when Docker is unavailable (`executor.md`). Security posture changes with no notification.
- `ResearchDirectorAgent` falls back from parallel to sequential execution if the parallel executor import fails (`research_director.md`).
- `get_experiment_cache()` ignores parameter changes after first call -- subsequent calls with different `similarity_threshold` return the same instance (`experiment_cache.md`).
## Domain Glossary

Terms extracted from module deep-read findings, defined in the context of the Kosmos codebase.

---

### Agent System

**BaseAgent**: Abstract base class (`kosmos/agents/base.py`) from which all research agents inherit. Provides lifecycle management (start/stop/pause/resume), message passing (sync and async), and state persistence. Five concrete subclasses: ResearchDirectorAgent, HypothesisGeneratorAgent, ExperimentDesignerAgent, DataAnalystAgent, LiteratureAnalyzerAgent.

**AgentMessage**: Pydantic model for inter-agent communication. Contains `from_agent`, `to_agent`, `content` (dict), `type` (MessageType), `correlation_id` for request-response pairing, and auto-generated UUID. The primary data contract for the message-passing system.

**AgentStatus**: Eight-value lifecycle enum: CREATED, STARTING, RUNNING, IDLE, WORKING, PAUSED, STOPPED, ERROR. Determines which lifecycle transitions are permitted.

**AgentState**: Pydantic snapshot model for agent persistence. Contains agent_id, agent_type, status, data dict, and timestamps. Used by `get_state()`/`restore_state()` for state preservation.

**AgentRegistry**: Central registry (`kosmos/agents/registry.py`) that stores `Dict[str, BaseAgent]` instances. Responsible for calling `set_message_router()` on each agent to wire up inter-agent message delivery. Provides `get_system_health()` aggregating status from all registered agents.

**MessageType**: Four-value enum: REQUEST, RESPONSE, NOTIFICATION, ERROR. Determines how messages are routed and how error responses are generated.

**Message Router**: Callback function set on each agent by the AgentRegistry. When an agent calls `send_message()`, the router delivers the message to the target agent's `receive_message()` method. If the router is not set, message delivery silently fails.

**ResearchDirectorAgent**: The master orchestrator agent (2953 lines). Coordinates the full autonomous research cycle. Contains both message-based async coordination and direct agent calls. The central hub through which all research workflow state flows.

### Research Workflow

**WorkflowState**: Nine-state enum defining the autonomous research lifecycle: INITIALIZING, GENERATING_HYPOTHESES, DESIGNING_EXPERIMENTS, EXECUTING, ANALYZING, REFINING, CONVERGED, PAUSED, ERROR. Each state has a defined set of allowed transitions.

**NextAction**: Eight-value enum specifying what the ResearchDirector should do next: GENERATE_HYPOTHESIS, DESIGN_EXPERIMENT, EXECUTE_EXPERIMENT, ANALYZE_RESULT, REFINE_HYPOTHESIS, CONVERGE, PAUSE, ERROR_RECOVERY.

**ResearchWorkflow**: State machine engine (`kosmos/core/workflow.py`) that enforces allowed transitions and maintains transition history. The ALLOWED_TRANSITIONS dict defines a directed graph of permitted state changes. PAUSED is a universal resumption state (can go to 6 states). CONVERGED is nearly terminal (only exit is back to GENERATING_HYPOTHESES).

**ResearchPlan**: Pydantic model tracking the state of an autonomous research session. Contains the research question, hypothesis pools (untested, tested, supported, rejected), experiment queues, iteration count, and convergence status. The structural backbone of the iteration tracking system.

**WorkflowTransition**: Pydantic record of a single state change, including from/to states, action description, timestamp, and metadata. Stored in the ResearchWorkflow's transition_history list.

**Convergence**: The determination that a research question has been sufficiently answered. Reached when hypotheses are supported/rejected with sufficient evidence. The ConvergenceDetector utility class (not a registered agent) evaluates convergence criteria including multiple comparison correction (FDR).

**Iteration**: One complete cycle through the research loop: generate hypothesis -> design experiment -> execute -> analyze -> refine. Bounded by `max_iterations` (default 10) and `MAX_ACTIONS_PER_ITERATION` (50, infinite loop guard).

### Hypothesis Pipeline

**Hypothesis**: Pydantic model (`kosmos/models/hypothesis.py`) representing a testable scientific statement. Required fields: research_question, statement (10-500 chars), rationale (min 20 chars), domain. Carries optional scores: testability_score, novelty_score, confidence_score, priority_score (all 0.0-1.0).

**HypothesisStatus**: Six-value lifecycle enum: GENERATED, UNDER_REVIEW, TESTING, SUPPORTED, REJECTED, INCONCLUSIVE. Default for new hypotheses is GENERATED.

**ExperimentType**: Three-value enum defining how a hypothesis can be tested: COMPUTATIONAL, DATA_ANALYSIS, LITERATURE_SYNTHESIS. Imported from `kosmos.models.hypothesis` and used across experiment design modules.

**HypothesisGenerationRequest**: Input model for the hypothesis generation pipeline. Specifies the research question, number of hypotheses to generate (default 3), and quality thresholds (min_novelty_score=0.5).

**HypothesisGenerationResponse**: Output model from hypothesis generation. Contains a list of Hypothesis objects, generation metadata, and convenience methods like `get_best_hypothesis()` (returns highest priority_score, or first if none scored).

**NoveltyReport**: Assessment of how novel a hypothesis is relative to existing knowledge. Contains novelty_score (0-1), max_similarity to known work, and prior_art_detected flag. Default novelty_threshold_used=0.75 (inconsistent with other thresholds in the system).

**TestabilityReport**: Assessment of whether and how a hypothesis can be tested. Contains testability_score (0-1), primary_experiment_type, resource estimates, challenges, and limitations. Default testability_threshold_used=0.3.

**PrioritizedHypothesis**: Wraps a Hypothesis with weighted scoring: novelty=0.30, feasibility=0.25, impact=0.25, testability=0.20. `update_hypothesis_priority()` computes and sets the hypothesis's priority_score.

**Hypothesis Evolution**: Phase 7 feature tracking hypothesis refinement lineage. Fields `parent_hypothesis_id`, `generation` (1 = original, 2+ = refined), `refinement_count`, and `evolution_history` enable tree-structured hypothesis exploration.

### Experiment System

**ExperimentProtocol**: The central Pydantic model (`kosmos/models/experiment.py`) defining an experiment's design. Contains name, hypothesis_id, experiment_type, steps (sequential ProtocolSteps), variables, control_groups, statistical_tests, sample_size (capped at 100,000), resource_requirements, and rigor_score (0-1).

**ProtocolStep**: Individual step in an experiment protocol. Has step_number (must be contiguous 1..N), title, description, action, dependencies (requires_steps), resource requirements, expected duration, and optional code_template.

**Variable**: Typed experiment variable with name, type (INDEPENDENT, DEPENDENT, CONTROL, CONFOUNDING), description, possible values, fixed_value, unit, and measurement_method.

**ControlGroup**: Experiment control group definition with name, description, variables dict, rationale, and sample_size (capped at 100,000).

**StatisticalTest / StatisticalTestSpec**: Enum of supported tests (T_TEST, ANOVA, CHI_SQUARE, CORRELATION, REGRESSION, MANN_WHITNEY, KRUSKAL_WALLIS, WILCOXON, CUSTOM) and the specification model for configuring a test with null hypothesis, alternative, alpha level, and required power.

**ResourceRequirements**: Model specifying what an experiment needs: compute hours, memory, GPU, estimated cost, API calls, duration, data sources, required libraries, and parallelization capability.

**ExperimentDesignRequest / ExperimentDesignResponse**: Input/output models for the experiment designer agent. The request specifies constraints (max cost, max duration, minimum rigor). The response contains the designed protocol plus validation results and feasibility assessment.

**ValidationReport**: Quality assessment of an experiment protocol covering rigor score, control group adequacy, sample size assessment, statistical power, bias detection, and reproducibility scoring.

**Rigor Score**: A 0-1 float assigned to experiment protocols indicating overall scientific rigor. Accounts for control groups, sample size, statistical power, and bias mitigation. Default minimum threshold is 0.6.

### Execution Engine

**CodeExecutor**: The code execution engine (`kosmos/execution/executor.py`). Executes LLM-generated Python (and R) code with safety sandboxing, output capture, retry logic, and optional profiling. Two execution paths: Docker sandbox (preferred) and restricted-builtins direct `exec()` (fallback).

**ExecutionResult**: Plain data class holding execution outcome: success status, return value, stdout/stderr captures, error info, execution time, profile data, and data_source indicator ('file' or 'synthetic').

**SAFE_BUILTINS**: Whitelist of ~70 Python builtins permitted in the restricted exec() environment. Excludes `open`, `input`, `getattr`, and `__import__` (replaced with restricted version). Security-critical configuration.

**_ALLOWED_MODULES**: Whitelist restricting imports to scientific/data libraries (numpy, pandas, scipy, sklearn) plus safe stdlib modules. The restricted import function checks only the top-level module name.

**RetryStrategy**: Self-correcting code repair engine handling 10+ error types. Attempts LLM-assisted repair first (limited to first 2 attempts), then falls back to pattern-based fixes (wrapping in try/except, auto-importing common modules). FileNotFoundError and SyntaxError are terminal (non-retryable).

**Docker Sandbox**: Preferred execution environment using Docker containers for isolation. When unavailable, the system silently falls back to restricted `exec()` -- callers are not notified of this downgrade.

**Return Value Convention**: Executed code must set a variable named `results` (plural, preferred) or `result` (singular) for outputs to be captured. This is a silent contract not validated or documented to the code generator.

**Data Source**: Indicator on ExecutionResult distinguishing 'file' (real data used) from 'synthetic' (generated data used). Set by the executed code itself.

### Result Analysis

**ExperimentResult**: The central Pydantic model (`kosmos/models/result.py`) for experiment outcomes. Contains status, execution metadata, raw/processed data, variable results, statistical tests, and the critical `supports_hypothesis` tri-state (True/False/None).

**ResultStatus**: Five-value enum: SUCCESS, FAILED, PARTIAL, TIMEOUT, ERROR. FAILED means the experiment ran but produced no useful results; ERROR means a system failure prevented execution.

**StatisticalTestResult**: Output of a statistical test with test statistic, p-value (bounded 0-1), effect size, confidence intervals, and three significance flags at alpha 0.05, 0.01, and 0.001. Significance labels: "***" (p<0.001), "**" (p<0.01), "*" (p<0.05), "ns" (not significant).

**supports_hypothesis**: Tri-state field on ExperimentResult. `True` = evidence supports the hypothesis, `False` = evidence contradicts it, `None` = no determination possible (e.g., no p-value available). Drives the ResearchDirector's decision to mark hypotheses as supported or rejected.

**Multiple Comparison Correction (FDR)**: Benjamini-Hochberg False Discovery Rate correction applied by the ResearchDirector to p-values before convergence checks. Prevents Type I error inflation across multiple hypothesis tests within a research session.

**ResultExport**: Wrapper supporting JSON, CSV (requires pandas), and Markdown export formats for experiment results.

### LLM Provider System

**LLMProvider**: Abstract base class (`kosmos/core/providers/base.py`) for all LLM integrations. Defines four abstract methods: `generate`, `generate_async`, `generate_with_messages`, `generate_structured`. Three concrete implementations: AnthropicProvider, OpenAIProvider, LiteLLMProvider.

**LLMResponse**: Core return type for generation methods. Contains `content` (str), `usage` (UsageStats), and `model`. Implements 25 string-delegation methods for backward compatibility (strip, split, find, etc.), but these return plain `str`, losing metadata. `isinstance(response, str)` returns `False`.

**UsageStats**: Token accounting dataclass: input_tokens, output_tokens, total_tokens (NOT auto-computed), optional cost_usd, model, provider, timestamp. The total_tokens field can be inconsistent across providers.

**ProviderAPIError**: Custom exception with provider name, message, status_code, raw_error, and recoverable flag. The `is_recoverable()` method classifies errors as retryable or terminal using status codes (4xx except 429 = non-recoverable) and message pattern matching (9 recoverable patterns checked before 10 non-recoverable patterns -- retry-favoring bias).

**Message**: Simple dataclass for conversation messages with role (bare str, not validated), content, optional name, and optional metadata. The `role` field accepts any string; validation happens only at the provider API boundary.

**ClaudeClient**: Legacy Anthropic-only LLM client in `kosmos/core/llm.py`. Being superseded by the provider-agnostic system but still in use. Contains response caching, auto model selection, and usage tracking.

**CLI Mode**: Special operational mode where the API key consists entirely of '9' characters. Bypasses actual API calls when running under Claude Code CLI. Detected by `api_key.replace('9', '') == ''`.

**ModelComplexity**: Heuristic classifier that estimates prompt complexity (0-100 score) based on token count and keyword matches. Only two recommendations: "haiku" (score < 30) or "sonnet" (score >= 30). Uses 20 complex keywords ('hypothesis', 'experiment', 'algorithm', etc.) biased toward research terminology.

**Auto Model Selection**: Feature where `ClaudeClient.generate()` automatically chooses between Haiku (cheaper, faster) and Sonnet (more capable) based on `ModelComplexity` scoring. Not available in `generate_with_messages()` or the provider system.

**Provider Registry**: Module-level dict in `factory.py` mapping provider names to classes. Registered at import time: anthropic, claude (alias), openai, litellm, ollama/deepseek/lmstudio (aliases for LiteLLM).

### Literature System

**PaperMetadata**: Dataclass (not Pydantic) in `kosmos/literature/base_client.py` representing a normalized academic paper. Contains identifiers (id, doi, arxiv_id, pubmed_id), citation metrics, content fields, and a priority chain for `primary_identifier` (DOI > arXiv > PubMed > source ID).

**PaperSource**: Five-value enum: ARXIV, SEMANTIC_SCHOLAR, PUBMED, UNKNOWN, MANUAL. The MANUAL source enables papers added by hand, not fetched from any API.

**BaseLiteratureClient**: Abstract base class defining the contract for all literature API clients. Four abstract methods: `search()`, `get_paper_by_id()`, `get_paper_references()`, `get_paper_citations()`. Concrete implementations: arXiv, Semantic Scholar, PubMed.

### Knowledge Graph

**KnowledgeGraph**: Neo4j-based persistence layer (`kosmos/knowledge/graph.py`) for scientific knowledge. Manages four node types (Paper, Concept, Method, Author) and five relationship types (CITES, USES_METHOD, DISCUSSES, AUTHORED, RELATED_TO). Uses the `py2neo` community library.

**Neo4jWorldModel**: Adapter class (`kosmos/world_model/simple.py`) wrapping KnowledgeGraph to implement the WorldModelStorage and EntityManager interfaces. Uses type-dispatch to route standard types to KnowledgeGraph methods and custom types to generic Cypher queries.

**Entity**: World model data class representing a node in the knowledge graph. Has an `id`, `type` (Paper, Concept, Author, Method, or custom), `properties` dict, `annotations` list, and metadata (confidence, project, verified status).

**Relationship**: World model data class representing an edge in the knowledge graph. Has source_id, target_id, type (CITES, USES_METHOD, DISCUSSES, AUTHORED, RELATED_TO, or custom), and properties dict.

**Cypher Injection**: Security vulnerability in `world_model/simple.py` where entity types, relationship types, and project names are injected into Cypher queries via f-strings without sanitization.

### Configuration

**KosmosConfig**: Pydantic `BaseSettings` singleton (`kosmos/config.py`) imported by 40 files. Contains typed subsections for every subsystem: LLM providers (claude, anthropic, openai, litellm), database, Redis, logging, literature APIs, vector DB, Neo4j, safety, performance, monitoring, development, and world model.

**get_config()**: Singleton accessor for KosmosConfig. Creates instance from environment variables and `.env` file on first call. Calls `create_directories()` to ensure log and data directories exist.

**LLM Provider Selection**: The `llm_provider` field on KosmosConfig selects which LLM backend is active: "anthropic" (default), "openai", or "litellm". The `validate_provider_config` model validator enforces that the selected provider's API key is present.

**Optional Provider Configs**: The `config.openai`, `config.claude`, and `config.anthropic` fields can be `None` if their respective API keys are not set. Accessing attributes on a None config raises `AttributeError`.

### Safety System

**SafetyGuardrails**: Central safety coordinator (`kosmos/safety/guardrails.py`) implementing emergency stop, resource limits, code validation, and incident logging. **Not currently wired into the execution pipeline** -- CodeExecutor uses CodeValidator directly, bypassing emergency stop checks.

**Emergency Stop**: Multi-source halt mechanism. Can be triggered by: Unix signals (SIGTERM/SIGINT), filesystem flag file (`.kosmos_emergency_stop`), or programmatic call. When triggered: sets in-memory status, creates flag file, kills Docker sandbox containers, logs safety incident.

**STOP_FLAG_FILE**: Relative path (`.kosmos_emergency_stop`) used for cross-process emergency stop signaling. An external process can trigger emergency stop by creating this file. The relative path means the check depends on the current working directory.

**CodeValidator**: Validates LLM-generated code before execution. Initialized by SafetyGuardrails with hardcoded permissions: allow_file_read=True, allow_file_write=False, allow_network=False. Used directly by CodeExecutor (bypassing SafetyGuardrails).

**SafetyIncident**: Pydantic model recording safety events. Logged to both in-memory list and JSONL file. Emergency stop incidents have `violation=None`.

**Resource Limits**: Caps on experiment execution resources. Default deny-by-default: max_memory_mb=2048, max_execution_time=300s, no network, no file write, no subprocess. Boolean permissions use AND logic -- since defaults are all False, requested permissions are always blocked.

### Database Layer

**get_session()**: Context manager yielding a SQLAlchemy Session. Auto-commits on clean exit, auto-rolls back on exception. Raises `RuntimeError` if database not initialized.

**init_database()**: Creates SQLAlchemy engine with database-type-specific configuration. For SQLite: `check_same_thread=False`. For PostgreSQL/MySQL: `QueuePool` with `pool_pre_ping=True`. Always calls `Base.metadata.create_all()`.

**init_from_config()**: High-level initialization from KosmosConfig. Runs first_time_setup (creates .env, runs migrations), calls init_database, verifies schema completeness. Degraded operation: setup errors logged as warnings but don't block initialization.

**Eager Loading**: SQLAlchemy `joinedload()` strategy used in get_experiment (default with_hypothesis=True) to prevent N+1 queries. Selectively enabled per-query.

**Slow Query Logging**: Event listener pattern on the SQLAlchemy engine that records query start time on `before_cursor_execute` and logs warnings for queries exceeding a configurable threshold (default 100ms). Batch operations (`executemany`) are excluded.

### Monitoring

**AlertManager**: Central manager for alert rules, active alerts, and notification dispatch (`kosmos/monitoring/alerts.py`). Evaluates 7 default rules, 3 of which are placeholders. Supports 4 notification channels: logging (always on), email/SMTP, Slack webhook, PagerDuty Events API v2.

**AlertRule**: Defines a trigger condition with name, callable condition, severity, message template, and cooldown period (default 5 minutes). If the condition callable raises, it is silently treated as "no alert needed".

**Alert ID**: Auto-generated as `"{name}_{unix_timestamp}"`. Same-name alerts triggered within one second share an ID, causing silent overwrites in the active alerts dict.

### Caching

**ExperimentCache**: Standalone SQLite-backed cache (`kosmos/core/experiment_cache.py`) for experiment results. Uses SHA256 fingerprinting for exact match and cosine similarity for approximate reuse detection. Separate from the general CacheManager system. Not currently imported by any production code.

**Fingerprint**: SHA256 hash of normalized hypothesis text + sorted parameter JSON. Used as the cache key for exact-match experiment lookup. Generated by `ExperimentNormalizer.generate_fingerprint()`.

**Similarity Search**: Linear scan over all cached experiment embeddings computing cosine similarity. No vector index. Returns entries above a configurable threshold (default 0.90).

**ClaudeCache**: Response cache used by `ClaudeClient` for LLM call deduplication. Cache key includes system prompt, max_tokens, temperature, and stop_sequences. Same prompt with different parameters misses cache.

### Plan Review

**PlanReviewerAgent**: Quality gate (`kosmos/orchestration/plan_reviewer.py`) that validates research plans on 5 dimensions (specificity, relevance, novelty, coverage, feasibility) before execution. Uses LLM-based scoring with mock fallback. Sits between PlanCreatorAgent and DelegationManager.

**PlanReview**: Dataclass containing review results: approved (bool), dimension scores (dict), average/min scores, feedback text, required changes, and suggestions.

**Structural Requirements**: Three checks on a research plan's tasks: at least 3 data_analysis tasks, at least 2 distinct task types, every task has description and expected_output.

**Mock Review**: Fallback when LLM is unavailable. Approves structurally valid plans with score 7.5, rejects invalid ones with score 6.0. Always appends a note identifying itself as a mock review.

**DIMENSION_WEIGHTS**: Dead code defining weights for 5 dimensions (specificity 0.25, relevance 0.25, novelty 0.20, coverage 0.15, feasibility 0.15). Scoring actually uses unweighted average.

### Logging

**JSONFormatter**: Custom logging formatter (`kosmos/core/logging.py`) producing structured JSON log records with UTC timestamps, correlation IDs, and optional workflow context fields (workflow_id, cycle, task_id).

**TextFormatter**: Human-readable log formatter with optional ANSI colors. Has a known side effect: mutates `record.levelname` in-place, which can leak ANSI codes to other handlers on the same logger.

**Correlation ID**: ContextVar-based request tracing identifier. Must be manually set by calling code -- no automatic propagation. Read by JSONFormatter. Thread-pool executors do not propagate it unless explicitly wrapped.

**ExperimentLogger**: Standalone utility tracking experiment lifecycle events (start, hypothesis, design, execution, result, error, end) via structured logging. Events are in-memory only with no persistence.

### Cross-Cutting Patterns

**Singleton Pattern**: Used extensively throughout Kosmos for shared state: `get_config()`, `get_client()`, `get_alert_manager()`, `get_experiment_cache()`, `get_knowledge_graph()`. Most are NOT thread-safe on first access.

**Dual Model System**: Pydantic models for runtime validation/API contracts and SQLAlchemy models for persistence. Conversion between the two is manual. Applies to Hypothesis, Experiment, and Result entities.

**Soft-Fail / Graceful Degradation**: Pattern where dependencies are optional and failures are silently swallowed. Examples: Neo4j disconnect (methods return defaults), Docker unavailability (sandbox silently downgraded), config import failures in hot paths (silently caught).

**Lazy Import**: Pattern of importing modules inside function bodies to avoid circular dependencies or defer expensive initialization. Used in: `send_message()` (imports config), `transition_to()` (imports config), `get_client()` (imports provider), ResearchDirector (imports all sub-agents).

**CLI Mode Detection**: Convention where API key consisting entirely of '9' characters triggers a special mode bypassing actual API calls. Used in ClaudeClient, AnthropicProvider, and ClaudeConfig.

**`datetime.utcnow()` Deprecation**: Used throughout the codebase (BaseAgent, WorkflowTransition, ResearchPlan, Hypothesis, Experiment, Result, JSONFormatter, ExperimentCache, WorldModel). Deprecated in Python 3.12+. Should be replaced with `datetime.now(timezone.utc)`.

**`use_enum_values=False`**: Pydantic ConfigDict setting used by Hypothesis, Experiment, and WorkflowTransition models. Means `.model_dump()` returns enum objects, not string values. Manual `to_dict()` methods call `.value` explicitly. This creates a split where `to_dict()` and `model_dump()` produce different output.

**Pydantic v1/v2 Compatibility**: The `kosmos.utils.compat.model_to_dict` helper abstracts differences between Pydantic versions. Used by result.py, guardrails.py, and research_director.py for serialization.
## Configuration Surface (see Module Index: config.py; see Change Impact Index: Hub 1)

### Configuration Architecture

Kosmos uses a **Pydantic BaseSettings singleton** as its central configuration mechanism [`kosmos/config.py:978-983`]. The master class `KosmosConfig` aggregates 16 typed sub-configurations, each a `BaseSettings` subclass [`kosmos/config.py:922-976`]. A module-level singleton stored as `_config: Optional[KosmosConfig] = None` is accessed via `get_config(reload=False)`, which provides lazy initialization [`kosmos/config.py:1136-1154`]. The `reload` parameter allows runtime re-read from environment but does **not** notify already-initialized components (LLM clients, DB connections).

### Configuration Hierarchy (Priority Order)

1. **Explicit constructor arguments** (e.g., `api_key=` in `ClaudeClient.__init__`)
2. **Environment variables** (e.g., `ANTHROPIC_API_KEY`, `LLM_PROVIDER`)
3. **`.env` file** at project root (loaded automatically by Pydantic via `SettingsConfigDict(env_file=str(Path(__file__).parent.parent / ".env"), case_sensitive=False, extra="ignore")` [`kosmos/config.py:978-983`])
4. **Field defaults** in Pydantic model definitions

Before any config loading, `from dotenv import load_dotenv; load_dotenv()` is called at module level in the CLI entry point [`kosmos/cli/main.py:19-22`], ensuring environment variables are populated before Pydantic reads them.

### Environment Variables

#### Tier 1: LLM Provider Keys (Required for Core Functionality)

| Variable | Default | Required | Location |
|----------|---------|----------|----------|
| `ANTHROPIC_API_KEY` | none | Yes (if `LLM_PROVIDER=anthropic`) | `kosmos/config.py:37-39`, `kosmos/core/llm.py:160-165`, `kosmos/core/providers/anthropic.py:88-92` |
| `OPENAI_API_KEY` | none | Yes (if `LLM_PROVIDER=openai`) | `kosmos/config.py:91`, `kosmos/core/providers/openai.py:32-130` |
| `LLM_PROVIDER` | `"anthropic"` | No | `kosmos/config.py:953-957`, `kosmos/cli/main.py:304` |
| `CLAUDE_MODEL` | `"claude-sonnet-4-5"` | No | `kosmos/config.py:29` |
| `OPENAI_MODEL` | (see config) | No | `kosmos/config.py:91` |

`ClaudeConfig.api_key` has alias `ANTHROPIC_API_KEY` and is a required field with no default; without it, config validation fails with "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic" [`kosmos/config.py:37-39`, `kosmos/config.py:1033-1038`]. The `validate_provider_config()` model validator enforces that the selected provider has a valid API key [`kosmos/config.py:1024-1043`]. LiteLLM validation is lenient since local models do not need keys.

`ClaudeClient.__init__()` reads `self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')` and raises `ValueError` if neither is provided [`kosmos/core/llm.py:160-165`]. `AnthropicProvider.__init__()` similarly reads `self.api_key = config.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')`, raising ValueError with instructions for both API and CLI modes [`kosmos/core/providers/anthropic.py:88-92`].

#### Tier 2: Database and Infrastructure (Has Defaults)

| Variable | Default | Required | Location |
|----------|---------|----------|----------|
| `DATABASE_URL` | `"sqlite:///kosmos.db"` | No | `kosmos/config.py:253-257` |
| `REDIS_URL` | `"redis://localhost:6379/0"` | No | `kosmos/config.py:305-317` |
| `REDIS_ENABLED` | `"false"` | No | `kosmos/config.py:305-317` |
| `REDIS_MAX_CONNECTIONS` | `50` | No | `kosmos/config.py:305-317` |

#### Tier 3: Optional Service Keys

| Variable | Default | Required | Location |
|----------|---------|----------|----------|
| `NEO4J_URI` | `"bolt://localhost:7687"` | No | `.env.example:298-309` |
| `NEO4J_USER` | `"neo4j"` | No | `.env.example:298-309` |
| `NEO4J_PASSWORD` | `"kosmos-password"` | No | `.env.example:298-309` |
| `SEMANTIC_SCHOLAR_API_KEY` | none | No | `.env.example:262-264` |
| `PUBMED_API_KEY` | none | No | `.env.example:262-264` |
| `PUBMED_EMAIL` | none | No | `.env.example:262-264` |
| `KOSMOS_SKILLS_DIR` | none | No | `kosmos/agents/skill_loader.py:148-149` |

Literature API keys are optional but increase rate limits when set [`.env.example:262-264`]. `KOSMOS_SKILLS_DIR` is read directly from `os.environ` to override the default skills directory path and is **not** modeled in any config class [`kosmos/agents/skill_loader.py:148-149`].

#### Tier 4: Monitoring and Alerting (All Optional, All Default to Disabled)

| Variable | Default | Required | Location |
|----------|---------|----------|----------|
| `ALERT_EMAIL_ENABLED` | `"false"` | No | `kosmos/monitoring/alerts.py:362-393` |
| `ALERT_EMAIL_FROM` | none | No | `kosmos/monitoring/alerts.py:362-393` |
| `ALERT_EMAIL_TO` | none | No | `kosmos/monitoring/alerts.py:362-393` |
| `SMTP_HOST` | `"localhost"` | No | `kosmos/monitoring/alerts.py:362-393` |
| `SMTP_PORT` | `"587"` | No | `kosmos/monitoring/alerts.py:362-393` |
| `SMTP_USER` | none | No | `kosmos/monitoring/alerts.py:362-393` |
| `SMTP_PASSWORD` | none | No | `kosmos/monitoring/alerts.py:362-393` |
| `ALERT_SLACK_ENABLED` | `"false"` | No | `kosmos/monitoring/alerts.py:415-421` |
| `SLACK_WEBHOOK_URL` | none | No | `kosmos/monitoring/alerts.py:415-421` |
| `ALERT_PAGERDUTY_ENABLED` | `"false"` | No | `kosmos/monitoring/alerts.py:483-493` |
| `PAGERDUTY_INTEGRATION_KEY` | none | No | `kosmos/monitoring/alerts.py:483-493` |

All 12+ monitoring env vars are read via raw `os.getenv()` calls and are **not** modeled in the Pydantic config [`kosmos/monitoring/alerts.py:362-493`].

#### Tier 5: LiteLLM Environment Variables

| Variable | Default | Required | Location |
|----------|---------|----------|----------|
| `LITELLM_MODEL` | `"gpt-3.5-turbo"` | No | `kosmos/core/providers/factory.py:162-169`, `kosmos/config.py:143` |
| `LITELLM_API_KEY` | none | No | `kosmos/core/providers/factory.py:162-169`, `kosmos/config.py:143` |
| `LITELLM_API_BASE` | none | No | `kosmos/core/providers/factory.py:162-169`, `kosmos/config.py:143` |
| `LITELLM_MAX_TOKENS` | (see config) | No | `kosmos/core/providers/factory.py:162-169` |
| `LITELLM_TEMPERATURE` | (see config) | No | `kosmos/core/providers/factory.py:162-169` |
| `LITELLM_TIMEOUT` | (see config) | No | `kosmos/core/providers/factory.py:162-169` |

A `sync_litellm_env_vars()` model validator manually syncs `LITELLM_*` env vars into the nested `LiteLLMConfig` because Pydantic nested `BaseSettings` do not automatically inherit parent `.env` values [`kosmos/config.py:986-1022`]. The factory also reads `LITELLM_*` directly when no config object is available [`kosmos/core/providers/factory.py:162-169`].

### Sub-Configuration Classes

Each sub-config is a standalone `BaseSettings` with its own `SettingsConfigDict(populate_by_name=True)` [`kosmos/config.py:922-976`]:

| Class | File:Line | Key Env Vars | Purpose |
|-------|-----------|-------------|---------|
| `ClaudeConfig` | `config.py:29` | `ANTHROPIC_API_KEY`, `CLAUDE_MODEL`, `CLAUDE_MAX_TOKENS` | Anthropic LLM settings |
| `OpenAIConfig` | `config.py:91` | `OPENAI_API_KEY`, `OPENAI_MODEL` | OpenAI provider settings |
| `LiteLLMConfig` | `config.py:143` | `LITELLM_MODEL`, `LITELLM_API_KEY`, `LITELLM_API_BASE` | Multi-provider proxy |
| `ResearchConfig` | `config.py:200` | `MAX_ITERATIONS`, `MAX_BUDGET_USD` | Research cycle limits |
| `DatabaseConfig` | `config.py:250` | `DATABASE_URL` | SQLAlchemy connection |
| `RedisConfig` | `config.py:305` | `REDIS_URL`, `REDIS_ENABLED` | Caching layer |
| `SafetyConfig` | `config.py:573` | `ENABLE_GUARDRAILS`, `SANDBOX_TYPE` | Safety/guardrail toggles |
| `Neo4jConfig` | `config.py:534` | `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Graph database |

Sub-config factory functions (`_optional_openai_config`, `_optional_anthropic_config`, `_optional_claude_config`) check `os.getenv()` before instantiating; if `OPENAI_API_KEY` is not set, the `openai` sub-config is `None` [`kosmos/config.py:896-919`]. This means `config.claude`, `config.openai`, and `config.anthropic` can all be `None`, requiring null checks at every call site. The `validate_provider_config` validator only checks the *active* provider [`kosmos/config.py:1024-1043`].

### Config Files

| File | Purpose | Notes |
|------|---------|-------|
| `.env` | Primary runtime configuration | Loaded by both `load_dotenv()` and Pydantic `SettingsConfigDict`. Created from `.env.example` on first run [`kosmos/utils/setup.py:17-49`]. |
| `.env.example` | Configuration template | Copied to `.env` by `ensure_env_file()` during `init_from_config()` [`kosmos/db/__init__.py:169`]. Contains default values including `NEO4J_PASSWORD=kosmos-password`. |
| `kosmos/config.py` | Config class definitions | All 16 sub-configuration classes, validators, and the singleton accessor. |

The CLI `config show` command inspects both `.env` and `.env.example` files to display configuration status [`kosmos/cli/commands/config.py:97-108`].

### Feature Flags

| Flag | Env Var | Default | Effect |
|------|---------|---------|--------|
| Redis caching | `REDIS_ENABLED` | `"false"` | Enables/disables the Redis caching layer [`kosmos/config.py:305-317`] |
| Safety guardrails | `ENABLE_GUARDRAILS` | (see config) | Toggles safety/guardrail system [`kosmos/config.py:573`] |
| Email alerts | `ALERT_EMAIL_ENABLED` | `"false"` | Enables email notification for alerts [`kosmos/monitoring/alerts.py:362-393`] |
| Slack alerts | `ALERT_SLACK_ENABLED` | `"false"` | Enables Slack webhook notifications [`kosmos/monitoring/alerts.py:415-421`] |
| PagerDuty alerts | `ALERT_PAGERDUTY_ENABLED` | `"false"` | Enables PagerDuty integration [`kosmos/monitoring/alerts.py:483-493`] |

### LLM Provider Configuration

#### Provider Selection

`KosmosConfig.llm_provider` is a `Literal["anthropic", "openai", "litellm"]` field with alias `LLM_PROVIDER`, defaulting to `"anthropic"` [`kosmos/config.py:953-957`].

#### Provider Architecture

The system implements a three-layer LLM abstraction [`kosmos/core/providers/base.py:156-334`]:

1. **Abstract base**: `LLMProvider(ABC)` defines four abstract methods: `generate()`, `generate_async()`, `generate_with_messages()`, `generate_structured()`. Optional `generate_stream()` and `generate_stream_async()` raise `NotImplementedError` by default.
2. **Concrete implementations**: `AnthropicProvider`, `OpenAIProvider`, `LiteLLMProvider`.
3. **Factory/registry**: `_PROVIDER_REGISTRY: Dict[str, type]` maps provider names to classes [`kosmos/core/providers/factory.py:16`].

`LLMResponse` dataclass wraps every provider response with `content`, `usage: UsageStats`, `model`, `finish_reason`, and `raw_response`. It implements 20+ string compatibility methods (`strip()`, `lower()`, `__contains__`, etc.) so callers expecting raw `str` responses work unchanged [`kosmos/core/providers/base.py:57-155`].

#### Provider Registry

`_register_builtin_providers()` runs at module import time and registers [`kosmos/core/providers/factory.py:189-217`]:

| Name | Provider Class | Notes |
|------|---------------|-------|
| `anthropic` | `AnthropicProvider` | Primary provider |
| `claude` | `AnthropicProvider` | Alias for backward compatibility |
| `openai` | `OpenAIProvider` | Official OpenAI, OpenRouter, Together AI, Ollama, LM Studio [`kosmos/core/providers/openai.py:32-130`] |
| `litellm` | `LiteLLMProvider` | 100+ backends via litellm library [`kosmos/core/providers/litellm_provider.py:40-120`] |
| `ollama` | `LiteLLMProvider` | Routed through LiteLLM |
| `deepseek` | `LiteLLMProvider` | Routed through LiteLLM |
| `lmstudio` | `LiteLLMProvider` | Routed through LiteLLM |

Each registration is wrapped in `try/except ImportError` for graceful degradation when packages are absent [`kosmos/core/providers/factory.py:189-217`].

#### Anthropic Provider Details

`AnthropicProvider` is the most complete implementation with sync/async generation, response caching via `ClaudeCache`, auto model selection (Haiku vs Sonnet) based on `ModelComplexity.estimate_complexity()`, streaming with event bus integration, and cost tracking [`kosmos/core/providers/anthropic.py:36-882`].

CLI mode detection: `self.is_cli_mode = self.api_key.replace('9', '') == ''` -- an API key of all 9s routes to Claude Code CLI proxy instead of the Anthropic API [`kosmos/core/providers/anthropic.py:109-110`].

Auto model selection: when enabled, uses `ModelComplexity.estimate_complexity(prompt, system)` to choose between Haiku (simple, score < 30) and Sonnet (moderate/complex). Only active in API mode, not CLI mode [`kosmos/core/providers/anthropic.py:182-203`].

#### Client Singleton

`_default_client` is a module-level singleton protected by `_client_lock = threading.Lock()` with double-checked locking for thread safety [`kosmos/core/llm.py:608-611`]. `get_client()` first tries the provider system via `get_provider_from_config(config)`, and on failure falls back to instantiating `AnthropicProvider` with env var config [`kosmos/core/llm.py:613-679`]. `get_provider()` is the recommended entry point for new code -- it calls `get_client(use_provider_system=True)` and asserts the result is an `LLMProvider` instance [`kosmos/core/llm.py:682-706`].

#### Retry and Error Handling

`ProviderAPIError` includes `recoverable: bool` and an `is_recoverable()` method that classifies errors by status code: 429 (rate limit) and 5xx are recoverable; 4xx (except 429) are not [`kosmos/core/providers/base.py:417-484`].

Async retry uses `tenacity` with `stop_after_attempt(3)`, `wait_exponential(multiplier=1, min=2, max=30)`, and a custom `should_retry` predicate. Non-recoverable errors (auth, JSON parse) are not retried [`kosmos/core/async_llm.py:440-459`].

Sync `ClaudeClient.generate_json()` implements its own retry loop: `for attempt in range(max_retries + 1)` with cache bypass on retries for JSON parse failures [`kosmos/core/llm.py:417-477`].

### Behavior on Missing Configuration

If `init_from_config()` fails (often due to missing API key), the CLI prints a user-friendly error listing possible causes but does **not** exit -- commands like `--help` and `version` still work [`kosmos/cli/main.py:144-165`].

Async LLM client init checks `api_key = os.getenv("ANTHROPIC_API_KEY")` and only initializes if truthy; if missing, it logs a warning and falls back to sequential LLM calls [`kosmos/agents/research_director.py:228-237`].

If `REDIS_ENABLED=true` but `REDIS_URL` is unreachable, the app discovers this at runtime, not at startup -- there is no validation for optional services at config load time.

### First-Time Setup

`ensure_env_file()` copies `.env.example` to `.env` on first run if missing [`kosmos/utils/setup.py:17-49`]. This is called during `init_from_config()` in `kosmos/db/__init__.py:169`.

### Safety: Environment Filtering

For reproducibility snapshots, only a safe-listed subset of env vars is captured: `env_vars = {k: v for k, v in os.environ.items() if k in safe_vars}` -- secrets are excluded [`kosmos/safety/reproducibility.py:208`].

### Dual-Path Configuration Hazards

Despite the centralized Pydantic config, many modules bypass it and read environment variables directly via `os.getenv()`:

- **Monitoring subsystem**: `kosmos/monitoring/alerts.py:362-493` uses 12+ raw `os.getenv()` calls for alerting configuration. None of these are modeled in the Pydantic config.
- **Health checks**: `kosmos/api/health.py:226-338` reads `REDIS_ENABLED`, `REDIS_URL`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` directly instead of using `get_config().redis` or `get_config().neo4j`.
- **LLM client**: `kosmos/core/llm.py:160` reads `os.environ.get('ANTHROPIC_API_KEY')` as fallback, duplicating what the Pydantic config already validates.
- **Provider factory**: `kosmos/core/providers/factory.py:164-168` reads `LITELLM_*` directly, despite these being synced via the model validator.
- **CLI doctor**: `kosmos/cli/main.py:304` reads `LLM_PROVIDER` directly via `os.getenv()`.
- **Skill loader**: `kosmos/agents/skill_loader.py:148-149` reads `KOSMOS_SKILLS_DIR` directly, not modeled in any config class.

This dual-path pattern creates two sources of truth. The monitoring subsystem is the worst offender with no Pydantic modeling at all. Pydantic nested models also do not auto-inherit `.env` values (the `sync_litellm_env_vars` workaround addresses this for LiteLLM only [`kosmos/config.py:985-1022`]; other sub-configs may silently miss `.env` values).
## Conventions

### Naming

**Always use snake_case for functions, methods, variables, and module names.** [PATTERN: universal] Classes use PascalCase consistently: `HypothesisGeneratorAgent` [`kosmos/agents/hypothesis_generator.py:33`], `DockerSandbox` [`kosmos/execution/sandbox.py:66`], `MaterialsOptimizer` [`kosmos/domains/materials/optimization.py:50`], `EventBus` [`kosmos/core/event_bus.py:28`]. Functions and methods use snake_case: `generate_hypotheses`, `execute_code`, `correlation_analysis` [`kosmos/agents/hypothesis_generator.py`, `kosmos/execution/sandbox.py`, `kosmos/domains/materials/optimization.py`].

**Always use UPPER_SNAKE_CASE for module-level constants.** [PATTERN: universal] Examples: `_DEFAULT_CLAUDE_SONNET_MODEL` [`kosmos/config.py:17`], `DEFAULT_IMAGE`, `DEFAULT_CPU_LIMIT`, `DEFAULT_MEMORY_LIMIT` [`kosmos/execution/sandbox.py:79-84`]. Private constants use a leading underscore prefix.

**Always prefix private/internal helper methods with a single underscore.** [PATTERN: universal] Examples: `_validate_json_dict`, `_validate_json_list` [`kosmos/db/operations.py:27-44`], `_on_start`, `_on_stop` [`kosmos/agents/base.py:169-187`], `_initialize_metabolic_pathways` [`kosmos/domains/biology/ontology.py:88`], `_parse_kegg_entry` [`kosmos/domains/biology/apis.py:91`]. No double-underscore name mangling is used anywhere.

**Always name test classes as `Test{Component}{Aspect}`.** [PATTERN: universal] Examples: `TestHypothesisGeneratorInit`, `TestHypothesisGeneration`, `TestHypothesisValidation` [`tests/unit/agents/test_hypothesis_generator.py:45, 71, 120`], `TestEventBusBasics`, `TestPublishSync`, `TestPublishAsync` [`tests/unit/core/test_event_bus.py:38, 77, 127`], `TestSandboxInitialization`, `TestImageVerification`, `TestSandboxExecution` [`tests/unit/execution/test_sandbox.py:49, 100, 134`].

**Always name test methods as `test_{action}_{condition_or_variant}` and include a one-line docstring.** [PATTERN: universal] Examples: `test_init_default` / "Test default initialization" [`tests/unit/agents/test_hypothesis_generator.py:48-49`], `test_sandbox_init_docker_unavailable` / "Test sandbox initialization when Docker unavailable" [`tests/unit/execution/test_sandbox.py:86-87`], `test_publish_sync_exception_continues` / "publish_sync continues after callback exception" [`tests/unit/core/test_event_bus.py:114-115`].

**Always name custom exceptions as `{Context}Error`.** [PATTERN: 6/6 observed] Examples: `PDFExtractionError` [`kosmos/literature/pdf_extractor.py:22`], `LiteratureCacheError` [`kosmos/literature/cache.py:19`], `CacheError` [`kosmos/core/cache.py:22`], `BudgetExceededError` [`kosmos/core/metrics.py:63`], `JSONParseError` [`kosmos/core/utils/json_parser.py:21`], `ProviderAPIError` [`kosmos/core/providers/base.py:417`].

### Imports

**Always group imports into three blocks separated by blank lines: (1) stdlib, (2) third-party, (3) kosmos-internal. Use absolute imports with the `kosmos.` prefix.** [PATTERN: universal] Example from `kosmos/agents/hypothesis_generator.py` [`lines 7-28`]: stdlib (`logging`, `time`, `uuid`, `datetime`), then internal via `from kosmos.agents.base import BaseAgent`, `from kosmos.core.llm import get_client`, `from kosmos.models.hypothesis import Hypothesis`. Similarly `kosmos/core/llm.py` [`lines 14-36`] separates stdlib from internal imports.

**Always use `from X import Y` style for specific symbols. Never use wildcard imports.** [PATTERN: universal] No `import *` patterns exist. All imports are explicit: `from kosmos.core.events import BaseEvent, EventType, StreamingEvent` [`kosmos/core/event_bus.py:14-18`], `from sqlalchemy import Column, String, Integer, Float, Boolean, Text, DateTime, ForeignKey, JSON` [`kosmos/db/models.py:12`].

**Always handle optional dependencies with `try/except ImportError` guards.** [PATTERN: consistent across providers] The LLM module demonstrates: `try: from anthropic import Anthropic ... except ImportError: HAS_ANTHROPIC = False` [`kosmos/core/llm.py:24-31`]. Provider registrations are each wrapped in `try/except ImportError` for graceful degradation [`kosmos/core/providers/factory.py:189-217`].

### Data Modeling

**Always use Pydantic BaseModel for runtime models with validation, field constraints, and serialization.** [PATTERN: 24+ modules] Examples: `Hypothesis` [`kosmos/models/hypothesis.py:32`], `DomainClassification` [`kosmos/models/domain.py:38`], `AgentMessage` [`kosmos/agents/base.py:45`], `WorkflowTransition` [`kosmos/core/workflow.py:45`], `BiologicalConcept` [`kosmos/domains/biology/ontology.py:44`], `CorrelationResult` [`kosmos/domains/materials/optimization.py:50`].

**Always use SQLAlchemy declarative Base for database-persisted models.** [PATTERN: concentrated in `kosmos/db/models.py`] The `Experiment`, `Hypothesis`, `Result` models inherit from `declarative_base()` [`kosmos/db/models.py:19-72`].

**Always use `@dataclass` for simple internal data containers and API response types that need no validation.** [PATTERN: 19 usages / 10 files] Examples: `KEGGPathway`, `GWASVariant`, `eQTLData` [`kosmos/domains/biology/apis.py:27-56`], `MaterialProperties`, `NomadEntry`, `AflowMaterial` [`kosmos/domains/materials/apis.py:43-80`], `Message`, `UsageStats`, `LLMResponse` [`kosmos/core/providers/base.py:18-77`].

**Always use `(str, Enum)` for all enum types to ensure JSON serialization compatibility.** [PATTERN: 6/6 observed] Every enum inherits from both `str` and `Enum`: `AgentStatus(str, Enum)` [`kosmos/agents/base.py:25`], `ExperimentStatus(str, enum.Enum)` [`kosmos/db/models.py:22`], `WorkflowState(str, Enum)` [`kosmos/core/workflow.py:18`], `ScientificDomain(str, Enum)` [`kosmos/models/domain.py:15`], `EvidenceLevel(str, Enum)` [`kosmos/domains/biology/genomics.py:50`], `BiologicalRelationType(str, Enum)` [`kosmos/domains/biology/ontology.py:31`].

**Always use Pydantic `Field()` with `description` for model attributes. Include numeric constraints where applicable.** [PATTERN: consistent across Pydantic models] Examples: `primary_domain: ScientificDomain = Field(description="The primary scientific domain identified")` [`kosmos/models/domain.py:42-44`], `statement: str = Field(..., min_length=10, max_length=500, description="Clear, testable hypothesis statement")` [`kosmos/models/hypothesis.py:52`], `confidence: float = Field(default=1.0, ge=0.0, le=1.0)` [`kosmos/domains/biology/ontology.py:63`].

### Type Annotations

**Always annotate all function signatures with type hints. Use `typing` module types for collections.** [PATTERN: universal] Examples: `def create_hypothesis(session: Session, id: str, ...) -> Hypothesis` [`kosmos/db/operations.py:87-98`], `def get_basic_health(self) -> Dict[str, Any]` [`kosmos/api/health.py:36`], `def subscribe(self, callback: Callback, event_types: Optional[List[EventType]] = None, ...) -> None` [`kosmos/core/event_bus.py:58-63`].

### Configuration

**Always use Pydantic BaseSettings for environment-based configuration with field aliases mapping to environment variable names.** [PATTERN: 16 sub-configs in `kosmos/config.py`] Use `model_config = SettingsConfigDict(populate_by_name=True)` for backward compatibility [`kosmos/config.py:8-84`]. Field validators use `@field_validator` and `@model_validator` decorators.

### Docstrings

**Always start every module with a module-level docstring describing its purpose and key components.** [PATTERN: universal] Examples: `kosmos/agents/base.py` [`lines 1-10`] describes purpose and async architecture. `kosmos/execution/__init__.py` [`lines 1-39`] lists components with usage examples. `kosmos/domains/biology/ontology.py` [`lines 1-24`] includes concept categories and usage example.

**Always use Google-style docstrings for classes and methods with Args/Returns/Raises sections.** [PATTERN: universal] Class docstrings list capabilities with bullet points: `BaseAgent` [`kosmos/agents/base.py:98-111`] lists "Provides:" and "Subclasses should implement:" sections. Method docstrings: `subscribe()` [`kosmos/core/event_bus.py:63-84`], `estimate_complexity()` [`kosmos/core/llm.py:57-65`].

**Always state explicit coverage targets in test file module docstrings.** [PATTERN: consistent across domain tests] Examples: "Coverage target: 30 tests across 5 test classes" [`tests/unit/domains/biology/test_ontology.py:6`], "Coverage target: 50 tests (10 clients x 5 tests each)" [`tests/unit/domains/biology/test_apis.py:6`], "Coverage target: 35 tests across 5 test classes" [`tests/unit/domains/materials/test_optimization.py:11`].

### Logging

**Always use `logger = logging.getLogger(__name__)` at module level. Never use `print()` for operational output.** [PATTERN: 112/110 files] This exact pattern appears in 110+ modules [`grep count returns 112 occurrences across 110 files`]. The logger is assigned immediately after imports, before any class or function definitions. Examples: `kosmos/core/llm.py:38`, `kosmos/db/operations.py:24`, `kosmos/execution/sandbox.py:15`, `kosmos/agents/base.py:22`. Log messages include context: `logger.info(f"Initialized HypothesisGeneratorAgent {self.agent_id}")` [`kosmos/agents/hypothesis_generator.py:89`].

### Error Handling

**Always define domain-specific exception classes inheriting from `Exception`. Catch at method boundaries and log before re-raising or returning `None`.** [PATTERN: consistent] Custom exceptions follow `{Domain}Error` naming (see Naming section above).

**Always use the catch-log-return-None pattern for API client methods.** [PATTERN: consistent across domain API clients] `KEGGClient.get_compound()` [`kosmos/domains/biology/apis.py:84-95`]: `except (httpx.HTTPError, RetryError, Exception) as e: logger.error(...); return None`. This is the standard across all domain API clients.

**Always log errors then re-raise for lifecycle methods.** [PATTERN: consistent in agent system] `BaseAgent.start()` [`kosmos/agents/base.py:168-175`]: `except Exception as e: self.status = AgentStatus.ERROR; logger.error(...); raise`.

**Always classify LLM errors by recoverability. Retry 429 and 5xx. Never retry 4xx (except 429).** [PATTERN: consistent across sync/async] `ProviderAPIError.is_recoverable()` classifies by status code [`kosmos/core/providers/base.py:417-484`]. Async retry: `tenacity` with 3 attempts, exponential backoff (min=2, max=30) [`kosmos/core/async_llm.py:440-459`]. Sync retry: manual loop in `generate_json()` with cache bypass on JSON parse failures [`kosmos/core/llm.py:417-477`].

### Async Patterns

**Always use async/await for I/O-bound operations (LLM calls, container management). Provide sync wrappers for backward compatibility.** [PATTERN: consistent across agent/execution layers] `BaseAgent` [`kosmos/agents/base.py:1-10`] documents that `send_message()`, `receive_message()`, `process_message()` are async with sync wrappers. `ProductionExecutor` [`kosmos/execution/production_executor.py:63`] uses `async def execute_code()`. `EventBus` supports both `publish_sync` and `publish` (async) [`kosmos/core/event_bus.py:28-46`].

### Class Structure

**Always organize behavioral classes with explicit method section banners using `# ========` separators.** [PATTERN: consistent in core classes] `BaseAgent` [`kosmos/agents/base.py:97-198`] organizes under `# LIFECYCLE MANAGEMENT`, `# MESSAGE HANDLING`. `DockerSandbox` [`kosmos/execution/sandbox.py:66-114`] follows the same approach. `operations.py` [`lines 47-86`] uses `# ============================================================================` banners for `# HYPOTHESIS CRUD`, `# QUERY PERFORMANCE MONITORING`.

### Testing

**Always mirror source tree structure: `kosmos/X/Y.py` maps to `tests/unit/X/test_Y.py`.** [PATTERN: 172 test files] The test tree replicates the package hierarchy exactly: `kosmos/agents/hypothesis_generator.py` -> `tests/unit/agents/test_hypothesis_generator.py` [`tests/unit/agents/test_hypothesis_generator.py:1`], `kosmos/core/event_bus.py` -> `tests/unit/core/test_event_bus.py` [`tests/unit/core/test_event_bus.py:1`], `kosmos/execution/sandbox.py` -> `tests/unit/execution/test_sandbox.py` [`tests/unit/execution/test_sandbox.py:1`].

**Always place shared fixtures in `tests/conftest.py`. Use tier-specific conftest files for environment setup.** [PATTERN: 3 conftest tiers] Root conftest [`tests/conftest.py:1-167`] provides session-scoped path fixtures (`fixtures_dir`, `sample_papers_json`, etc.) and function-scoped temp fixtures (`temp_dir`, `temp_file`). Integration conftest [`tests/integration/conftest.py:11-33`] sets Neo4j and Anthropic env vars with `autouse=True` session-scoped fixture. E2e conftest [`tests/e2e/conftest.py:56-84`] defines reusable skip decorators (`requires_llm`, `requires_anthropic`, `requires_neo4j`, `requires_docker`, `requires_full_stack`).

**Always load `.env` at conftest import time before any fixtures are defined.** [PATTERN: 2/2 conftest files that need it] Both `tests/conftest.py` [`line 24-30`] and `tests/e2e/conftest.py` [`lines 17-19`] call `load_dotenv(override=True)` at module import time.

**Always provide pre-configured mock fixtures in root conftest for commonly-mocked external dependencies.** [PATTERN: 5 standard mocks] `mock_llm_client` (Claude LLM), `mock_anthropic_client` (raw Anthropic API), `mock_knowledge_graph` (Neo4j), `mock_vector_db` (ChromaDB), `mock_concept_extractor` (LLM-powered extraction) [`tests/conftest.py:174-245`].

**Always use `unittest.mock.Mock` for sync and `AsyncMock` for async interfaces. Target `@patch` at the import path in the module under test.** [PATTERN: universal] `@patch('kosmos.execution.sandbox.docker')` patches Docker at the import site, not globally [`tests/unit/execution/test_sandbox.py:52, 67, 85, 103-104`]. API tests: `with patch('httpx.Client', return_value=mock_httpx_client)` [`tests/unit/domains/biology/test_apis.py:47-48`].

**Always mark every test class with `@pytest.mark.unit`, `@pytest.mark.integration`, or `@pytest.mark.e2e`.** [PATTERN: universal] All unit test classes use `@pytest.mark.unit` [`tests/unit/agents/test_hypothesis_generator.py:44`, `tests/unit/core/test_event_bus.py:38`]. Use `@pytest.mark.asyncio` for async test methods [`tests/unit/core/test_event_bus.py:131, 141`].

**Always use `pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"))` for tests requiring real LLM calls.** [PATTERN: consistent] `pytestmark = [pytest.mark.requires_claude, pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Requires ANTHROPIC_API_KEY for real LLM calls")]` [`tests/unit/agents/test_hypothesis_generator.py:20-26`].

**Always define component-under-test fixtures locally in the test file, function-scoped by default. Use `autouse=True` for singleton reset fixtures.** [PATTERN: universal] Local fixtures: `hypothesis_agent` [`tests/unit/agents/test_hypothesis_generator.py:34-41`], `event_bus` [`tests/unit/core/test_event_bus.py:33-35`], `test_db` (in-memory SQLite) [`tests/unit/db/test_database.py:14-23`]. Singleton reset: `reset_event_bus()` before and after each test with `autouse=True` [`tests/unit/core/test_event_bus.py:24-29`].

**Always use `scope="session"` for expensive fixtures that persist across the entire test session.** [PATTERN: consistent in conftest] Session-scoped: `fixtures_dir`, `sample_papers_json`, `sample_arxiv_xml`, `sample_semantic_scholar_json`, `sample_pubmed_xml`, `sample_bibtex`, `sample_ris`, `sample_papers_data` [`tests/conftest.py:37-100`].

**Always use JSON fixture files in `tests/fixtures/` for complex test data. Use `np.random.seed(42)` for reproducible numeric data. Use `unique_id()` factory functions for test isolation.** [PATTERN: consistent] Fixture files: `fixtures_dir / "sample_papers.json"`, `fixtures_dir / "sample_arxiv_response.xml"` [`tests/conftest.py:43-64`]. Seeded random: `np.random.seed(42)` generating 60-row DataFrames [`tests/unit/domains/materials/test_optimization.py:33-71`]. Factory: `def unique_id() -> str: return uuid.uuid4().hex[:8]` [`tests/unit/agents/test_hypothesis_generator.py:29-31`].

### Domain Module Structure

**Always structure each scientific domain as a package under `kosmos/domains/{domain_name}/` containing `__init__.py`, `apis.py`, `ontology.py`, and one or more domain-specific analyzer modules.** [PATTERN: 3/3 implemented domains] Biology: `__init__.py`, `apis.py`, `genomics.py`, `metabolomics.py`, `ontology.py` [`kosmos/domains/biology/`]. Materials: `__init__.py`, `apis.py`, `ontology.py`, `optimization.py` [`kosmos/domains/materials/`]. Neuroscience: `__init__.py`, `apis.py`, `connectomics.py`, `neurodegeneration.py`, `ontology.py` [`kosmos/domains/neuroscience/`].

**Always re-export all public symbols from domain `__init__.py` with explicit `__all__`, grouped into labeled sections (API Clients, Data Models, Analyzers, Ontology).** [PATTERN: 3/3 implemented domains] `biology/__init__.py` [`lines 1-75`] imports 26 symbols. `materials/__init__.py` [`lines 1-58`] imports 19 symbols. `neuroscience/__init__.py` [`lines 1-71`] imports 19 symbols.

**Always define API clients in `apis.py` with `BASE_URL` class constant, `httpx.Client`, and `@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))` decorators.** [PATTERN: 2/2 fully implemented domains] `KEGGClient`: `BASE_URL = "https://rest.kegg.jp"`, `self.client = httpx.Client(timeout=timeout)` [`kosmos/domains/biology/apis.py:66-73`]. Materials APIs follow the same structure [`kosmos/domains/materials/apis.py:30-100`].

**Always define ontology classes with `concepts: Dict[str, Concept]` and `relations: List[Relation]` attributes, initialized via private `_initialize_*` methods.** [PATTERN: 3/3 implemented domains] `BiologyOntology.__init__` calls `_initialize_metabolic_pathways()`, `_initialize_genetic_concepts()`, `_initialize_disease_concepts()` [`kosmos/domains/biology/ontology.py:78-86`]. `MaterialsOntology` calls five `_initialize_*` methods [`kosmos/domains/materials/ontology.py:81-89`]. `NeuroscienceOntology` calls five `_initialize_*` methods [`kosmos/domains/neuroscience/ontology.py:49-59`].

**Always register new domains in the `ScientificDomain` enum in `kosmos/models/domain.py`.** [PATTERN: 8 enum values for 3 implementations + 2 stubs + 3 future] Current values: `BIOLOGY`, `NEUROSCIENCE`, `MATERIALS`, `PHYSICS`, `CHEMISTRY`, `ASTRONOMY`, `SOCIAL_SCIENCE`, `GENERAL` [`kosmos/models/domain.py:15-25`].

**Neuroscience may reuse biology types instead of defining its own ontology types.** [PATTERN: 1/3 -- neuroscience only] `NeuroscienceOntology` imports and reuses `BiologicalRelationType`, `BiologicalConcept`, `BiologicalRelation` from biology [`kosmos/domains/neuroscience/ontology.py:30-34`]. This is intentional: neuroscience shares biological relationship semantics (IS_A, PART_OF, REGULATES). Materials science has distinct relationship types (HAS_STRUCTURE, PROCESSED_BY, DOPED_WITH) justifying its own.

**Always use `model_config = ConfigDict(arbitrary_types_allowed=True)` in Pydantic models that hold numpy arrays or pandas DataFrames.** [PATTERN: 1/1 -- materials optimizer] `MaterialsOptimizer` [`kosmos/domains/materials/optimization.py:65`] uses this for fields holding numpy/pandas objects.

### Package Structure

**The `kosmos/domains/__init__.py` is intentionally empty -- domains are loaded on demand via `DomainRouter`, not imported wholesale.** [PATTERN: 1/1] [`kosmos/domains/__init__.py` is 1 line]. Individual domain `__init__.py` files provide full re-exports, but the parent package does not aggregate [`kosmos/core/domain_router.py` exists].

**The `kosmos/core/__init__.py` is intentionally empty -- the core package uses explicit imports rather than re-exporting.** [PATTERN: 1/1] Unlike `kosmos/agents/__init__.py` [`lines 1-29`] which provides a full public API, `core` requires callers to import specific modules [`kosmos/core/__init__.py` is 1 line].
## Gotchas

### Agent System

1. [CRITICAL] **Silent message routing failure** -- Agents are lazy-initialized and never registered in the message router. `send_message()` silently fails for unregistered agents (`kosmos/agents/research_director.py:1392-1397`). All action handlers now use direct calls instead. This was the root cause of Issue #76. (see Critical Path 0) (related: Gotcha #11 -- same root cause: message routing architecture) (related: Gotcha #12 -- same root cause: dual communication paths)

2. [HIGH] **Dual lock architecture with no enforcement** -- Both `asyncio.Lock` and `threading.RLock` exist side by side (`kosmos/agents/research_director.py:193-200`). Async code paths use `async with self._research_plan_lock`, sync code paths use `with self._research_plan_lock_sync`. Using the wrong lock type provides zero protection. No compile-time or runtime check prevents mixing. (see Critical Path 0) (related: Gotcha #6 -- same root cause: async/sync boundary)

3. [HIGH] **`_on_pause`/`_on_resume` hooks never fire** -- `pause()` and `resume()` do not call `_on_pause()` or `_on_resume()` despite defining them as override points (`kosmos/agents/base.py:189-205` vs `base.py:511-516`). Subclass authors implementing these hooks will never see them triggered.

4. [HIGH] **Sync wrappers deadlock if called from the event loop thread** -- `send_message_sync()` calls `run_coroutine_threadsafe(...).result(timeout=30)` when a loop is running (`kosmos/agents/base.py:318-322`). If called from the same thread as the event loop, this blocks forever because the coroutine cannot run while the thread is blocked waiting for its result.

5. [HIGH] **`execute()` signature contract broken by 2 subclasses** -- Base declares `execute(self, task: Dict[str, Any]) -> Dict[str, Any]` (`kosmos/agents/base.py:485`). `HypothesisGeneratorAgent.execute()` takes `AgentMessage` (`hypothesis_generator.py:91`). `ExperimentDesignerAgent.execute()` also takes `AgentMessage` (`experiment_designer.py:109`). Polymorphic dispatch through `BaseAgent` will pass wrong type.

6. [HIGH] **`_workflow_context()` yields without locking** -- The async-compatible version is a no-op context manager (`yield self.workflow`) providing no actual thread safety for async code paths (`kosmos/agents/research_director.py:377-379`).

7. [HIGH] **`time.sleep()` in async error recovery** -- `_handle_error_with_recovery()` uses blocking `time.sleep()` in async context, halting the event loop during backoff (`kosmos/agents/research_director.py:674`). The class uses `asyncio.Lock` for concurrent operations (`kosmos/agents/research_director.py:193`), so this blocks all concurrent work. (see Critical Path 0; see Error Handling Strategy) (related: Gotcha #2, #6 -- same root cause: async/sync boundary)

8. [HIGH] **ResearchDirectorAgent.process_message() is sync, not async** -- Base defines `async def process_message()` (`kosmos/agents/base.py:382`). `ResearchDirectorAgent` overrides with `def process_message()` (non-async) (`kosmos/agents/research_director.py:568`). Awaiting a non-coroutine is fragile and relies on CPython implementation detail.

9. [MEDIUM] **`register_message_handler()` is dead code** -- Stores handlers in `self.message_handlers` but nothing reads from it (`kosmos/agents/base.py:406-415`). Grep for `message_handlers` across all `*.py` shows only init and assignment, no dispatch.

10. [MEDIUM] **`_async_message_queue` is write-only** -- `receive_message()` puts messages into `_async_message_queue` (`kosmos/agents/base.py:338`) but no code anywhere calls `.get()` on it. The queue grows unbounded for the lifetime of the agent.

11. [MEDIUM] **AgentRegistry._route_message() silently drops messages** -- If the target agent is not found, the message is lost with a log warning (`kosmos/agents/registry.py:230-247`). No dead letter queue exists.

12. [MEDIUM] **Dual communication paths create maintenance burden** -- Coexistence of `_send_to_*` (message-based) and `_direct_*` (direct call) methods (`kosmos/agents/research_director.py:1039-1219` vs `1391-1979`) means every new agent integration must be wired into both systems.

13. [MEDIUM] **Deprecated `asyncio.get_event_loop().run_until_complete()`** -- Used in `execute_experiments_batch()` fallback (`kosmos/agents/research_director.py:2171`). Will warn or fail in Python 3.12+.

14. [MEDIUM] **Knowledge graph persistence failures silently swallowed** -- All `_persist_*_to_graph()` methods catch all exceptions and log warnings (`kosmos/agents/research_director.py:435-436, 474-475`). Graph data loss is invisible at runtime.

15. [MEDIUM] **Double database init** -- `init_from_config()` is called in both the CLI callback (`main.py:145`) and `ResearchDirectorAgent.__init__` (`research_director.py:131`). The second call's RuntimeError is caught only if message contains "already initialized" (`research_director.py:136`).

16. [MEDIUM] **Lazy-init agent latency** -- Agent slots (`_hypothesis_agent`, `_experiment_designer`, `_code_generator`, `_code_executor`, `_data_provider`, `_data_analyst`, `_hypothesis_refiner`) are all None at init (`kosmos/agents/research_director.py:145-151`). First execution of each phase has extra latency from agent construction.

17. [LOW] **`get_state()` has write side effect** -- Mutates `self.updated_at` (`kosmos/agents/base.py:446`). Callers expecting a pure read will inadvertently change agent state.

18. [LOW] **`start()` is one-shot, no restart** -- Guard checks `status != CREATED` (`kosmos/agents/base.py:161`). After `stop()`, agent cannot be restarted. Must create a new instance.

19. [LOW] **`restore_state()` does not restore queues or counters** -- Only restores 6 fields (`kosmos/agents/base.py:463-468`). `messages_sent`, `messages_received`, `tasks_completed`, `errors_encountered`, `message_queue`, `_message_router` are all lost on restore.

20. [LOW] **AgentRegistry header says "Not yet integrated"** -- Header says "Not yet integrated into the main research loop" despite being fully implemented (`kosmos/agents/registry.py:1-6`).

### Execution Engine

21. [HIGH] **Unrestricted exec() fallback** -- If Docker sandbox is unavailable (`SANDBOX_AVAILABLE=False`), code executes via `exec()` with restricted builtins (`kosmos/execution/executor.py:474-476, 589-597`). The restricted import allows 30+ modules including numpy, pandas, scipy (`executor.py:86-94`). No filesystem write restrictions beyond what the import whitelist prevents.

22. [HIGH] **Silent sandbox downgrade** -- If Docker is unavailable, `use_sandbox` is silently set to `False` (`kosmos/execution/executor.py:215-224`). Callers have no callback or exception for this security-relevant change. (see Critical Path 2) (related: Gotcha #28 -- same root cause: safety mechanisms not enforced in execution pipeline)

23. [HIGH] **Windows timeout leak** -- On Windows, `_exec_with_timeout` cannot kill the exec'd thread (`kosmos/execution/executor.py:622`). The thread continues running in background. Only the wait is abandoned.

24. [HIGH] **Return value convention undocumented to code generator** -- Executed code MUST set a variable named `results` or `result` for outputs to be captured (`kosmos/execution/executor.py:516`). This convention is not validated or communicated to the LLM code generator. All 5 templates follow this convention but LLM-generated code may not.

25. [HIGH] **Provenance pipeline not wired** -- 0/4 delegation routes in `DelegationManager._execute_task()` (`kosmos/orchestration/delegation.py:359-386`) attach `code_provenance` to returned findings. DelegationManager does not import or reference any provenance or notebook generation modules. The infrastructure exists but is disconnected from the live orchestration path.

26. [MEDIUM] **LLM repair prompt injection** -- `_repair_with_llm` sends user-generated code directly to an LLM prompt (`kosmos/execution/executor.py`). If the code contains adversarial content, it could manipulate the repair.

27. [MEDIUM] **Retry wrapping recursion** -- Pattern-based fixes wrap code in try/except. If retried again, the already-wrapped code gets wrapped again, producing nested try/except blocks (`kosmos/execution/executor.py:751-825`).

### Safety

28. [HIGH] **SafetyGuardrails NOT wired into execution pipeline** -- `SafetyGuardrails` is not used by `CodeExecutor` or `execute_protocol_code()`. `CodeExecutor` calls `CodeValidator` directly, bypassing the guardrails' emergency stop checks and resource limit enforcement (`kosmos/safety/guardrails.py`, `kosmos/execution/executor.py`). (see Critical Path 2) (related: Gotcha #22 -- same root cause: safety mechanisms not enforced in execution pipeline)

29. [HIGH] **`is_emergency_stop_active()` has write side effect** -- Checking status can TRIGGER emergency stop if the flag file exists (`kosmos/safety/guardrails.py:205-214`). A function named as a status check performs a state mutation.

30. [MEDIUM] **Relative flag file path** -- `STOP_FLAG_FILE` is relative to cwd (`kosmos/safety/guardrails.py:44`). If the process changes directory, flag file checks look in the wrong place.

31. [MEDIUM] **Resource limits are purely advisory** -- `enforce_resource_limits()` returns capped limits but does not enforce them at the OS level (`kosmos/safety/guardrails.py:142-178`). The caller must pass the limits to the sandbox configuration manually.

32. [MEDIUM] **Reset/flag file desync** -- If flag file removal fails during `reset_emergency_stop()`, the next status check will re-trigger the stop (`kosmos/safety/guardrails.py`).

33. [LOW] **Signal handler thread restriction** -- If instantiated from a non-main thread, signal handler registration silently fails, disabling the signal-based emergency stop pathway (`kosmos/safety/guardrails.py`).

34. [LOW] **Docker kill scope** -- Emergency stop kills ALL containers with `kosmos.sandbox=true` label, not just those from the current session (`kosmos/safety/guardrails.py`).

### LLM Integration

35. [HIGH] **`LLMResponse` is not `str`** -- `isinstance(response, str)` returns False despite `LLMResponse` subclassing `str` at the protocol level (`kosmos/core/providers/base.py:57-154`). String methods like `.strip()` return `str`, not `LLMResponse`, losing metadata. (see Module Index: kosmos/core/providers/base.py)

36. [HIGH] **No unit tests for provider base types** -- Zero automated tests exist for `LLMResponse`, `UsageStats`, `Message`, `ProviderAPIError`, and `LLMProvider` base class (`kosmos/core/providers/base.py`).

37. [HIGH] **Mock scores mask LLM outages** -- When the LLM is unreachable, `PlanReviewer` returns 5.0/10.0 scores silently (`kosmos/orchestration/plan_reviewer.py:160-162`). If `min_average_score` is set to 5.0 or below, all plans auto-approve during an outage. The `feedback` field says `"Failed to parse review"` (`plan_reviewer.py:267`) but this is buried in data, not surfaced as an alert.

38. [HIGH] **Circuit breaker and error recovery are independent** -- The `CircuitBreaker` in `async_llm.py` and `_handle_error_with_recovery` in `research_director.py` both track consecutive failures with threshold=3 but share no state (`async_llm.py:68` vs `research_director.py:45`). The research director could retry while the circuit breaker is open. (see Error Handling Strategy)

39. [MEDIUM] **`generate_with_messages()` ignores auto model selection** -- Always uses `self.model`, not the complexity-based selection (`kosmos/core/llm.py:389`). This is undocumented behavior.

40. [MEDIUM] **Usage counters not thread-safe** -- `LLMProvider` usage counters (`total_input_tokens`, `total_output_tokens`, `request_count`) are instance-level with no locking (`kosmos/core/providers/base.py:190-193`). `self.request_count += 1` is not atomic under concurrent async calls.

41. [MEDIUM] **`__iter__` yields characters** -- `LLMResponse.__iter__` yields single characters, not lines or tokens (`kosmos/core/providers/base.py:80-154`). Code iterating over a response gets individual chars.

42. [MEDIUM] **`total_tokens` not auto-computed** -- Some async paths leave `UsageStats.total_tokens` at 0 (`kosmos/core/providers/base.py:50`).

43. [MEDIUM] **`generate_structured()` returns Dict not LLMResponse** -- Usage stats only available via `get_usage_stats()`, not from the return value (`kosmos/core/providers/base.py:280-308`).

44. [MEDIUM] **`is_recoverable()` ordering bias** -- Recoverable patterns checked before non-recoverable patterns (`kosmos/core/providers/base.py:476-477`). Ambiguous messages like "invalid rate_limit response" default to recoverable.

45. [MEDIUM] **LLM caching is Anthropic-only** -- Response caching via `ClaudeCache` is implemented only in `AnthropicProvider` (`kosmos/core/providers/anthropic.py`). Switching providers silently loses caching.

46. [MEDIUM] **Streaming event bus integration is Anthropic-only** -- Only `AnthropicProvider` emits `LLMEvent` events to the event bus during streaming. Other providers lack this integration, so monitoring tied to the event bus will not work.

47. [MEDIUM] **No automatic provider failover** -- If the configured LLM provider fails, there is no automatic failover to an alternative. The only fallback is from config-based init to env-var-based `AnthropicProvider` init within `get_client()` (`kosmos/core/llm.py:613-679`).

48. [MEDIUM] **LLM parse failure defaults to 5.0** -- Borderline scores that almost always reject, creating silent quality degradation rather than an explicit error signal (`kosmos/orchestration/plan_reviewer.py:229-270`).

49. [MEDIUM] **LLM call logging silently disabled** -- Three providers (`anthropic.py:416-417`, `openai.py:356-357`, `core/workflow.py:297-298`) silently swallow config-loading exceptions with `except Exception: pass`. LLM call logging and workflow transition logging can be silently disabled with no diagnostic trail.

50. [LOW] **Cost estimation hardcodes "claude-sonnet-4-5"** even when other models are used (`kosmos/core/llm.py:519`). Savings calculations are rough approximations.

51. [LOW] **CLI mode detection is fragile** -- Any string consisting entirely of '9' characters matches, regardless of length (`kosmos/core/llm.py:179`, `kosmos/core/providers/anthropic.py:109-110`).

52. [LOW] **Qwen special-casing only in LiteLLM provider** -- Running Qwen through OpenAI-compatible endpoint misses no-think directive (`kosmos/core/providers/litellm_provider.py:157-168`).

53. [LOW] **`generate_stream_async` has dead code** -- Unreachable code paths in the async streaming implementation (`kosmos/core/providers/base.py`).

### Configuration

54. [HIGH] **`config.openai` can be `None`** -- Accessing attributes on it without checking raises `AttributeError` (`kosmos/config.py:962`). Same for `config.claude` and `config.anthropic` when their API keys are unset. The `validate_provider_config` validator only checks the active provider, so inactive provider configs are silently `None` (`kosmos/config.py:896-919`). (see Module Index: config.py; see Configuration Surface)

55. [HIGH] **Dual-path config reads** -- Health checks (`kosmos/api/health.py:226-338`), CLI doctor (`kosmos/cli/main.py:304`), monitoring (`kosmos/monitoring/alerts.py:362-493`), and provider factory (`kosmos/core/providers/factory.py:164-168`) all bypass the Pydantic config system entirely via raw `os.getenv()`. Changing a value in the Pydantic config object at runtime will NOT propagate to these modules. (see Critical Path 1; see Configuration Surface) (related: Gotcha #58 -- same root cause: env vars not modeled in Pydantic config) (related: Gotcha #109 -- same root cause: config bypass in health checks)

56. [HIGH] **No runtime config reload notification** -- `get_config(reload=True)` replaces the singleton but does not notify already-initialized components (LLM clients, DB connections) (`kosmos/config.py:1136-1154`). Components initialized before a reload continue using stale config.

57. [MEDIUM] **Default password in `.env.example`** -- `NEO4J_PASSWORD=kosmos-password` ships as a default (`.env.example:298-309`). `ensure_env_file()` (`kosmos/utils/setup.py:17-49`) automatically copies `.env.example` to `.env` without modification, so this insecure default can reach production.

58. [MEDIUM] **Monitoring env vars invisible to config system** -- All 12+ alerting variables (`ALERT_EMAIL_ENABLED`, `SMTP_HOST`, `SLACK_WEBHOOK_URL`, `PAGERDUTY_INTEGRATION_KEY`, etc.) at `kosmos/monitoring/alerts.py:362-493` are read via raw `os.getenv()` and are not modeled in any Pydantic config class. `config show` will not display them.

59. [MEDIUM] **`KOSMOS_SKILLS_DIR` is undocumented** -- This env var (`kosmos/agents/skill_loader.py:148-149`) overrides the skills directory path but is not modeled in any config class, not present in `.env.example`, and not shown by `config show`.

60. [MEDIUM] **Pydantic nested BaseSettings env var inheritance** -- Nested `BaseSettings` subclasses do not automatically inherit `.env` values from the parent. Only LiteLLM has a manual workaround (`sync_litellm_env_vars` at `kosmos/config.py:986-1022`). Other sub-configs may silently miss `.env` values and fall back to field defaults.

61. [MEDIUM] **Singleton ignores parameter changes** -- `get_experiment_cache()` uses first-call parameters; subsequent calls with different thresholds return the same instance with the original configuration (`kosmos/core/experiment_cache.py:729-751`).

62. [LOW] **Default model constants are module-level** -- `_DEFAULT_CLAUDE_SONNET_MODEL` and `_DEFAULT_CLAUDE_HAIKU_MODEL` (`kosmos/config.py:17-18`) are imported by `kosmos.models.hypothesis` (`hypothesis.py:13`). Changing them affects hypothesis metadata.

63. [LOW] **`ResearchConfig.max_runtime_hours` upper bound (24) contradicts documentation (12 hours)** -- The `max_runtime_hours` field allows up to 24.0 (`kosmos/config.py:241`) but the description mentions "up to 12 hours continuous operation" (`kosmos/config.py:243`).

64. [LOW] **Neo4j default password is hardcoded** -- `"kosmos-password"` appears in both config (`kosmos/config.py:549`) and health check (`kosmos/knowledge/graph.py:153`).

### Database

65. [HIGH] **Immediate commit in CRUD functions** -- Cannot batch operations atomically. Each `create_*` or `update_*` calls `session.commit()` (`kosmos/db/operations.py:113`) and `session.refresh()` (`operations.py:114`) independently. The `get_session()` context manager also calls `session.commit()` on clean exit (`kosmos/db/__init__.py:132`), leading to potential double-commits. (see Module Index: db.py; see Error Handling Strategy)

66. [MEDIUM] **`get_session()` rollback can mask exceptions** -- The rollback at line 134 is not wrapped in its own try/except (`kosmos/db/__init__.py:133-134`). If `session.rollback()` itself fails (e.g., connection lost), the original exception from the yield block is lost.

67. [MEDIUM] **Synchronous DB sessions in async methods** -- `get_session()` returns sync `Session`, blocking the event loop when called from async code. No async session support exists.

68. [MEDIUM] **Parameter name collision** -- `update_research_session` uses `db_session` (`kosmos/db/operations.py:572`) while all other functions use `session` as the parameter name.

69. [LOW] **SQLite lacks connection pooling** -- The `pool_size` parameter has no effect for SQLite databases, but no warning is issued (`kosmos/db/__init__.py:72-78`).

70. [LOW] **No soft-delete support** -- All models lack a `deleted_at` or `is_deleted` field. No `delete_*` functions are defined in `operations.py`. Deletion happens only via cascade (`kosmos/db/models.py:69`).

### Data Models

71. [HIGH] **`to_dict()` and `model_dump()` produce different output** -- Due to `use_enum_values=False` (`kosmos/models/experiment.py:575`). `to_dict()` manually converts enums to `.value`; `model_dump()` returns enum objects by default. This affects `Hypothesis` (`kosmos/models/hypothesis.py:156`), `ExperimentProtocol` (`experiment.py:575`), and `ExperimentResult` (`result.py`). Always use `to_dict()` for serialization that needs string enum values. (see Data Contracts)

72. [HIGH] **`export_markdown()` crashes on None stats** -- The `:.2f` format strings in the variable statistics table do not handle None values (`kosmos/models/result.py:350-353`). If `VariableResult` has None stats, `TypeError` is raised.

73. [HIGH] **`PaperSource.OTHER` does not exist** -- `citations.py` references `PaperSource.OTHER` at lines 206 and 262. The `PaperSource` enum in `base_client.py:17-23` only defines ARXIV, SEMANTIC_SCHOLAR, PUBMED, UNKNOWN, and MANUAL. Will raise `AttributeError` at runtime when parsing BibTeX or RIS files.

74. [MEDIUM] **Inconsistent novelty thresholds** -- `NoveltyReport.novelty_threshold_used=0.75` (`kosmos/models/hypothesis.py:259`) vs `HypothesisGenerationRequest.min_novelty_score=0.5` (`hypothesis.py:184`) vs `Hypothesis.is_novel(threshold=0.5)` (`hypothesis.py:150`). Three different defaults for "is this novel enough."

75. [MEDIUM] **`None` scores fail threshold checks** -- `is_testable()` and `is_novel()` return `False` when scores are `None` (`kosmos/models/hypothesis.py:146-147, 152-153`). A newly created hypothesis with no scores will fail both checks even if it is actually testable and novel.

76. [MEDIUM] **Step numbering must be contiguous** -- Gaps in step numbers (e.g., steps 1, 3, 5) will fail validation (`kosmos/models/experiment.py:435`). Expected numbers are `set(range(1, len(v) + 1))`.

77. [MEDIUM] **`statistical_tests` requires unique test_names** -- Duplicate test names cause validation failure (`kosmos/models/result.py:212-213`). If two tests of the same type are run, they must have distinct names.

78. [MEDIUM] **Pervasive use of deprecated `datetime.utcnow()`** -- Found across 10+ modules including `experiment.py:410-411`, `hypothesis.py:76-77`, `world_model/simple.py`, `core/logging.py:52`. Should use `datetime.now(timezone.utc)` for Python 3.12+ compatibility.

79. [LOW] **`get_best_hypothesis()` silently returns first hypothesis as fallback** -- When no hypothesis has a `priority_score` (`kosmos/models/hypothesis.py:229`), returns the first element regardless of quality.

80. [LOW] **pandas is a hidden dependency** -- `export_csv()` in `result.py` will fail at runtime if pandas is not installed (`kosmos/models/result.py:292`), but this is not declared in the model's imports.

81. [LOW] **`ensure_title` replaces bad titles silently** -- Replaces with "Untitled Step" (`kosmos/models/experiment.py:181`) rather than raising, masking LLM output quality issues.

### Knowledge Graph / World Model

82. [HIGH] **Counter increment bug** -- `create_authored`, `create_discusses`, and `create_uses_method` increment counters (paper_count, frequency, usage_count) on every call, even when the relationship already exists (`kosmos/knowledge/graph.py:614-616, 658-660, 703-705`). Repeated calls produce incorrect counts. (see Module Index: knowledge_graph.py; see Critical Path 3)

83. [HIGH] **Parameter mismatch bugs** -- `_add_author_entity` and `_add_method_entity` pass a `metadata` kwarg that `KnowledgeGraph` methods don't accept. `add_relationship` for CITES passes `paper_id` instead of `citing_paper_id` (`kosmos/world_model/simple.py:171-192, 482-488`). Would raise `TypeError` at runtime for those specific code paths.

84. [HIGH] **Cypher injection vectors** -- Entity type, relationship type, and project name are injected via f-strings into Cypher queries without sanitization (`kosmos/world_model/simple.py:231, 508, 684`). (see Module Index: world_model_simple.py)

85. [MEDIUM] **Silent connection failure** -- `KnowledgeGraph` constructor catches all connection errors and continues (`kosmos/knowledge/graph.py:80-99`). Code that does not check `.connected` will get `AttributeError` on `None` graph.

86. [MEDIUM] **World model fallback to ephemeral storage** -- `get_world_model()` silently falls back from Neo4j to `InMemoryWorldModel` (`kosmos/world_model/factory.py`). Callers may not realize they are writing to memory that will be lost on process exit.

87. [MEDIUM] **Export while disconnected writes empty file** -- `export_graph()` writes `{"entities": [], "relationships": []}` when Neo4j is disconnected rather than raising, potentially overwriting valid backup files (`kosmos/world_model/simple.py:670-676`).

88. [MEDIUM] **Inconsistent error contract** -- Some methods return `None` on not-found, others raise `ValueError` (`kosmos/world_model/simple.py`). No clear pattern.

89. [MEDIUM] **4-query paper lookup** -- `get_paper()` tries 4 separate queries sequentially for each identifier type (`kosmos/knowledge/graph.py`). For hot paths, this is a performance issue.

90. [LOW] **Non-thread-safe singleton** -- `get_knowledge_graph()` has a race condition on first access (`kosmos/knowledge/graph.py:999-1038`).

91. [LOW] **Cascading delete** -- `delete_paper()` and all delete methods remove ALL relationships of the deleted node with no confirmation (`kosmos/knowledge/graph.py`).

### Monitoring and Alerts

92. [HIGH] **No test coverage for alerts** -- Zero automated tests exist for the 569-line `kosmos/monitoring/alerts.py` module.

93. [MEDIUM] **Alert ID collisions** -- Same-name alerts triggered within one second share an ID, causing silent overwrites in `active_alerts` (`kosmos/monitoring/alerts.py:46-49`).

94. [MEDIUM] **No thread safety on AlertManager** -- No locking on `active_alerts` or `alert_history` mutation (`kosmos/monitoring/alerts.py`). Concurrent calls to `evaluate_rules()` and `resolve_alert()` could corrupt state.

95. [MEDIUM] **SMTP hardcoded to STARTTLS** -- Port 465 (implicit TLS) connections will fail; only port 587 pattern works (`kosmos/monitoring/alerts.py`).

96. [MEDIUM] **PagerDuty silently drops non-ERROR/CRITICAL** -- INFO and WARNING severities are filtered out without documentation (`kosmos/monitoring/alerts.py:487-488`).

97. [MEDIUM] **Singleton handler registration frozen at creation time** -- Changing env vars after first `get_alert_manager()` call has no effect (`kosmos/monitoring/alerts.py:528-552`).

98. [LOW] **3 of 7 default rules are placeholders** -- `high_api_failure_rate`, `api_rate_limit_warning`, and `high_experiment_failure_rate` always return False (`kosmos/monitoring/alerts.py:284-321`).

### Caching

99. [HIGH] **No test coverage for experiment cache** -- Zero automated tests exist for the 758-line `kosmos/core/experiment_cache.py` module.

100. [MEDIUM] **`total_experiments` counter inflates** -- `INSERT OR REPLACE` increments the stat even on overwrites (`kosmos/core/experiment_cache.py`).

101. [MEDIUM] **`_increment_stat()` bypasses the lock** -- Opens its own connection outside the class RLock, creating a thread-safety gap (`kosmos/core/experiment_cache.py`).

102. [MEDIUM] **`get_cached_result()` swallows all errors** -- Returns `None` on any exception with no logging (`kosmos/core/experiment_cache.py`).

103. [LOW] **Linear similarity scan** -- `find_similar()` loads all embeddings into memory with no indexing for scale (`kosmos/core/experiment_cache.py`).

104. [LOW] **`reset_experiment_cache()` does not close DB** -- The old instance's SQLite connection is abandoned and relies on GC (`kosmos/core/experiment_cache.py`).

### CLI and Startup

105. [HIGH] **`load_dotenv()` at import time** -- Importing `kosmos.cli.main` mutates `os.environ` from `.env` file (`kosmos/cli/main.py:19`). Test isolation risk for any test that imports the CLI module. (see Module Index: cli_main.py; see Configuration Surface)

106. [HIGH] **All-or-nothing command registration** -- One broken command module import disables all 7 commands silently. `register_commands()` swallows `ImportError` with `logging.debug` (`kosmos/cli/main.py:401-415`).

107. [MEDIUM] **Database init on every CLI invocation** -- Even `kosmos version` triggers database initialization attempt (`kosmos/cli/main.py:143-166`).

108. [MEDIUM] **No HTTP mounting for health endpoint** -- The health checker is a library, not a FastAPI route. No HTTP endpoint exists in the main app for `/health` or `/ready` (`kosmos/api/health.py`).

109. [MEDIUM] **Config bypass in all health checks** -- All four sub-checks read env vars directly via `os.getenv()` rather than through the Pydantic config system, creating potential divergence (`kosmos/api/health.py`).

110. [MEDIUM] **Warning blocks readiness** -- `_check_external_apis()` can return `"warning"` for non-standard API key formats. The readiness check treats warning as not-healthy (`kosmos/api/health.py`).

111. [LOW] **`logging.basicConfig` is effectively one-shot** -- Repeated calls in the same process are no-ops (`kosmos/cli/main.py`).

112. [LOW] **Hard-coded version in test** -- `test_cli.py:82` asserts `"v0.2.0"` -- will break on version bumps.

### Logging

113. [HIGH] **TextFormatter mutates record.levelname** -- ANSI escape codes are baked into the level name (`kosmos/core/logging.py:128`). If both a `TextFormatter` console handler and a `JSONFormatter` file handler are attached to the same logger, the JSON output may contain ANSI codes because the record object is shared. (see Module Index: core_logging.py)

114. [MEDIUM] **`datetime.utcfromtimestamp()` deprecation** -- Deprecated in Python 3.12 (`kosmos/core/logging.py:52`). Should be replaced with `datetime.fromtimestamp(record.created, tz=timezone.utc)`.

115. [MEDIUM] **ContextVar correlation_id has no auto-propagation** -- Must be manually set per request/task (`kosmos/core/logging.py:23-25`). `asyncio.create_task` copies context automatically, but thread-pool executors do not unless explicitly wrapped.

116. [LOW] **`ExperimentLogger` has no persistence** -- Events are in-memory only (`kosmos/core/logging.py:270`). If the process crashes, the experiment log is lost.

### Literature

117. [MEDIUM] **`_validate_query()` logs "truncating" but does NOT truncate** -- Callers that trust the validation may send overlong queries to APIs that have their own length limits (`kosmos/literature/base_client.py:249-250`).

118. [MEDIUM] **`_normalize_paper_metadata()` is not `@abstractmethod`** -- A subclass can forget to implement it and only discover the error at runtime when a search result is being processed (`kosmos/literature/base_client.py:255-268`).

119. [MEDIUM] **ArXiv citation data unavailable** -- `get_paper_references()` and `get_paper_citations()` both return empty lists (`kosmos/literature/arxiv_http_client.py:325-326, 343-344`). Citation data requires Semantic Scholar.

120. [MEDIUM] **PDF text mutation in place** -- `PDFExtractor.extract_paper_text()` stores result directly on `paper.full_text` (`kosmos/literature/pdf_extractor.py:213`) rather than returning a new object.

121. [MEDIUM] **LiteLLM provider event asymmetry** -- The LiteLLM provider does NOT emit any streaming events. Clients subscribed to LLM events will receive nothing when using LiteLLM.

122. [LOW] **`raw_data` excluded from `to_dict()`** -- Round-trip through `to_dict()` loses the raw API response (`kosmos/literature/base_client.py:99-122`).

123. [LOW] **`PaperMetadata.authors` type hint is `List[Author]` but defaults to `None`** -- Between construction and `__post_init__`, the field is `None`, not a list (`kosmos/literature/base_client.py:53`).

### Cross-Cutting

124. [HIGH] **38+ silent `except Exception: pass` sites** -- Worst offenders: `unified_search.py` (3 sites losing search results), `simple.py` (3 sites masking Neo4j problems), `base.py` (2 sites swallowing agent message logging failures).

125. [HIGH] **BudgetExceededError uncaught in orchestration** -- Raised only in `kosmos/core/metrics.py:801` with no catch site in the orchestration layer. Could crash the entire research loop. (see Error Handling Strategy)

126. [HIGH] **7 of 14 singletons have no reset function** -- Makes test isolation difficult for MetricsCollector, AlertManager, StageTracker, ExperimentCache, ClaudeCache, AgentRegistry, and HealthChecker. (see Shared State) (related: Gotcha #127 -- same root cause: singleton initialization deficiencies)

127. [HIGH] **Singleton creation races** -- Most `get_*()` factory functions use `if _X is None: _X = create()` without locking. `get_metrics_collector()`, `get_knowledge_graph()`, and 5+ other accessors can race under concurrent access.

128. [MEDIUM] **Dual message queues risk double-processing** -- Both sync list and async Queue receive the same messages; consumers reading the wrong queue may process messages twice.

129. [MEDIUM] **Unbounded WebSocket queue** -- Each WebSocket connection gets its own `asyncio.Queue` with no size limit configured. A slow client could accumulate unbounded memory.

130. [MEDIUM] **Sync-to-async event delivery gap** -- `publish_sync()` schedules async callbacks via `loop.create_task()` only if a running event loop exists. Otherwise silently skips them (`kosmos/core/event_bus.py:178-184`).

131. [MEDIUM] **No startup validation for optional services** -- If `REDIS_ENABLED=true` but `REDIS_URL` is unreachable, the app discovers this at runtime, not at startup.

132. [MEDIUM] **Chemistry and physics domains are stubs** -- Both contain only empty `__init__.py` files. Their enum values exist in `ScientificDomain` (`kosmos/models/domain.py:21-22`) but no APIs, ontology, or analyzers are implemented.

133. [MEDIUM] **Legacy `ClaudeClient` coexists with provider system** -- The old `ClaudeClient` class in `kosmos/core/llm.py` coexists alongside the provider system. `kosmos/core/providers/anthropic.py:881` aliases `ClaudeClient = AnthropicProvider`. New code should use `get_provider()`, not `get_client()`.

134. [MEDIUM] **`kosmos/core/__init__.py` is intentionally empty** -- Unlike `kosmos/agents/__init__.py` which provides a full public API, `core` requires explicit module-level imports. Do not expect `from kosmos.core import EventBus` to work.

135. [MEDIUM] **Dataclass vs Pydantic split** -- API response types use `@dataclass` (no validation), computed result models use Pydantic `BaseModel`. Example: `kosmos/domains/biology/apis.py:27-56` uses dataclasses, `kosmos/domains/biology/genomics.py:68` uses BaseModel. Do not add validation logic to dataclass response types.

### Testing

136. [MEDIUM] **Two parallel skip-condition systems** -- Root `tests/conftest.py` uses `skip_api_key = pytest.mark.skip(...)` as local variables (`tests/conftest.py:426-430`), while `tests/e2e/conftest.py` defines separate `requires_*` decorators (`tests/e2e/conftest.py:56-84`). These are independent systems that do not share logic.

137. [MEDIUM] **Integration tests lack shared DB fixture** -- The integration conftest only sets up Neo4j and Anthropic environment (`tests/integration/conftest.py`). Individual test files must handle their own database state, leading to potential test pollution.

138. [LOW] **No `reset` for singleton alert manager or DB state** -- Tests importing `app` get shared global state from prior test runs (`kosmos/cli/main.py`).

### Orchestration

139. [MEDIUM] **DIMENSION_WEIGHTS are dead code** -- Scoring uses unweighted average despite weights being defined (`kosmos/orchestration/plan_reviewer.py:68-75`).

140. [MEDIUM] **Docstring claims `required_skills` is checked** -- Implementation only checks `description` and `expected_output` (`kosmos/orchestration/plan_reviewer.py:14` vs `306-313`).

141. [MEDIUM] **Sync method tested with async** -- `review_plan()` is synchronous but test fixtures use `@pytest.mark.asyncio` and `await` (`kosmos/orchestration/plan_reviewer.py:101`).

142. [LOW] **Mock fallback always approves structurally valid plans** -- Returns 7.5 scores. No way to distinguish "LLM was down" from "LLM approved" without checking the `feedback` field for the "Mock review" string (`kosmos/orchestration/plan_reviewer.py:316-357`).

### Workflow

143. [MEDIUM] **Transition validation prevents skipping states** -- Cannot go directly from INITIALIZING to EXECUTING. The research cycle must follow the full state graph (`kosmos/core/workflow.py`).

144. [MEDIUM] **`reset()` is destructive** -- Clears all transition history with no undo (`kosmos/core/workflow.py`).

145. [MEDIUM] **Transition history grows unboundedly** -- `transition_history` has no maximum size and accumulates in memory for the entire session (`kosmos/core/workflow.py`).

146. [LOW] **Duration calculations are O(n^2)** -- `get_state_duration()` does a nested scan of `transition_history`. For long-running sessions, this could become slow (`kosmos/core/workflow.py`).
## Hazards -- Do Not Read

These files waste context tokens and should be avoided when working with an AI coding assistant. Read the suggested alternative instead.

### Virtual Environment

| Path | Reason | Size |
|------|--------|------|
| `venv/` | Third-party packages. Contains 500K+ lines of vendored pip, setuptools, and chardet code. Zero project logic. | ~500K lines |

### Generated and Compiled

| Path | Reason |
|------|--------|
| `**/__pycache__/` | Python bytecode cache. Regenerated automatically. |
| `kosmos_ai_scientist.egg-info/` | Package metadata generated by setuptools. Read `pyproject.toml` instead. |
| `htmlcov/` | HTML coverage report generated by pytest-cov. Read `coverage.xml` for raw data if needed. |
| `kosmos.db`, `kosmos_test.db` | SQLite database files. Binary, not readable. Read `kosmos/db/models.py` for schema. |
| `:memory:` | Artifact from SQLite in-memory testing. Empty file. |

### Large Data and Artifacts

| Path | Reason |
|------|--------|
| `data/benchmarks/paper_accuracy_benchmark.json` | 1484-line benchmark dataset. Reference data, not code. |
| `evaluation/personas/runs/*/` | Evaluation run artifacts with large JSON component files (168+ lines each). Outputs of evaluation runs, not source code. |
| `chroma_db/` | ChromaDB vector database storage. Binary index files. |
| `neo4j_data/`, `neo4j_logs/`, `neo4j_plugins/`, `neo4j_import/` | Neo4j database storage, logs, and plugins. Infrastructure artifacts, not code. |
| `postgres_data/` | PostgreSQL data directory. Infrastructure artifact. |
| `redis_data/` | Redis persistence files. Infrastructure artifact. |
| `logs/` | Runtime log output. Not source code. |
| `test_artifacts/` | Test output artifacts. Generated during test runs. |
| `kosmos-figures/` | Generated figure outputs. |
| `human_review_audit.jsonl` | Audit log file. Runtime output. |
| `.concept_extraction_cache/` | Cached concept extraction results. Regenerated automatically. |

### Archived and Historical

| Path | Reason |
|------|--------|
| `archive/` | Historical planning documents that may be outdated. The CLAUDE.md explicitly warns: "Don't read `archive/` for current architecture." |
| `archived/` | Archived checkpoints and bug-fix branches. Contains `CHECKPOINT_*.md` files, old issue tracking, and resolved bug-fix branches. Not current. |
| `docs/archive/` | Archived documentation including old implementation plans (`120525_implementation_gaps_v2.md`, `120625_code_review.md`), migration plans (`MIGRATION_MULTI_PROVIDER.md`, `MOCK_TESTS_MIGRATION_PLAN.md`), and contributor guides that may be stale. |
| `paper/` | Research paper drafts. Not code. |

### Large Files with Low Signal-to-Noise Ratio

| Path | Lines | Reason | Read Instead |
|------|-------|--------|--------------|
| `kosmos/agents/research_director.py` | 2952 | Largest source file. Monolithic orchestrator with 7 lazy-init agent slots, dual communication patterns, 60+ methods. Useful to understand but too large to read whole. | Read specific method ranges based on task. Key sections: `__init__` (68-260), `decide_next_action` (2388-2548), `execute` (2868-2909). |
| `kosmos/execution/data_analysis.py` | 1189 | Statistical analysis helper. Mostly boilerplate pandas/scipy wrappers. | Read only if modifying statistical methods. |
| `kosmos/workflow/ensemble.py` | 1141 | Ensemble workflow execution. | Read only if modifying multi-hypothesis parallel execution. |
| `kosmos/agents/literature_analyzer.py` | 1081 | Literature analysis agent. | Read `kosmos/literature/base_client.py` first for the interface, then specific methods as needed. |
| `kosmos/world_model/simple.py` | 1159 | Neo4j world model. Many methods with similar patterns. | Read `kosmos/world_model/interface.py` for the interface contract first. |
| `kosmos/config.py` | 1160 | 16 Pydantic sub-configs. Most are field definitions with defaults. | Read `/tmp/deep_crawl/sections/config_surface.md` for the curated summary. |

### Docker and Infrastructure Config

| Path | Reason |
|------|--------|
| `docker/` | Docker build files. Infrastructure, not application logic. |
| `k8s/` | Kubernetes manifests. Deployment config, not application logic. |
| `docker-compose.yml` | Service composition. Read only if debugging container orchestration. |
| `Dockerfile` | Build instructions. Read only if debugging image builds. |
| `alembic/versions/` | Database migration scripts (3 files). Read `kosmos/db/models.py` for current schema. The migrations show schema history but not current state. |
| `alembic.ini`, `alembic/env.py`, `alembic/script.py.mako` | Alembic configuration boilerplate. |

### Test Fixtures

| Path | Reason |
|------|--------|
| `tests/fixtures/sample_semantic_scholar_response.json` | 181-line mock API response. Reference data for tests. |
| `tests/fixtures/` (generally) | Test data files. Read only when debugging specific test failures. |

### External Skill Definitions

| Path | Reason |
|------|--------|
| `kosmos-claude-scientific-skills/` | External skill markdown files for LLM prompt injection. 116 skill directories. Not executable code. Read `kosmos/agents/skill_loader.py` to understand how they are loaded. |
| `kosmos-reference/` | Reference materials. Not code. |
| `prompts/` | Prompt templates. Read only when debugging LLM prompt construction. |
## Extension Points

### Adding a New Agent Type (see Change Playbook 1)

To add a new agent (e.g., PeerReviewerAgent), start at `kosmos/agents/base.py` for the `BaseAgent` abstract class and contract. Then:

1. Create `kosmos/agents/peer_reviewer.py` -- subclass `BaseAgent`, implement `execute(task: Dict) -> Dict` (`base.py:485-497`). Call `super().__init__(agent_id, agent_type, config)` as the first line. Set `self.status` to `WORKING`/`IDLE`/`ERROR` during execution.
2. Add import and `__all__` entry in `kosmos/agents/__init__.py` (`__init__.py:22-29`).
3. Add a lazy-init slot `self._peer_reviewer = None` in `kosmos/agents/research_director.py:145-152`.
4. Add a `_handle_peer_review_action()` method following the pattern at `research_director.py:1391-1979`.
5. Update `decide_next_action()` at `research_director.py:2388-2548` if the agent participates in the research loop.
6. Create models in `kosmos/models/` if the agent produces structured output (follow `kosmos/models/hypothesis.py` pattern).
7. Add DB models in `kosmos/db/models.py` and CRUD ops in `kosmos/db/operations.py` if data is persisted.
8. Write tests in `tests/unit/agents/test_peer_reviewer.py` (follow `tests/unit/agents/test_hypothesis_generator.py` pattern).

### Adding a New Scientific Domain (see Change Playbook 2)

To add a new domain (e.g., ecology), start at `kosmos/domains/biology/` as the reference implementation. Thirteen registration points must be updated:

1. Create `kosmos/domains/ecology/` with `__init__.py`, `apis.py`, `ontology.py`, and analyzer modules (follow `kosmos/domains/biology/` four-file pattern).
2. Add `'kosmos.experiments.templates.ecology'` to the `template_packages` list in `kosmos/experiments/templates/base.py:392-396`.
3. Add `"ecology"` to the `enabled_domains` default list in `kosmos/config.py:210-213`.
4. Add `"ecology"` to the `valid_domains` set in `kosmos/cli/commands/config.py:246`.
5. Add `"ecology"` to the `default_domains` list in `kosmos/agents/research_director.py:265`.
6. Add skill bundle and domain mapping in `kosmos/agents/skill_loader.py:46-82` and `91-102`.
7. Add domain default experiment type in `kosmos/agents/experiment_designer.py:394-404`.
8. Create experiment templates under `kosmos/experiments/templates/ecology/` (extend `TemplateBase` from `base.py:98-336`).

### Adding a New LLM Provider

To add a new provider (e.g., Google Gemini), start at `kosmos/core/providers/base.py` for the `LLMProvider` abstract class.

1. Create `kosmos/core/providers/gemini.py` -- subclass `LLMProvider` (`base.py:155-415`). Implement `generate()`, `generate_stream()`, `generate_structured()`, and their async variants.
2. Register in `kosmos/core/providers/factory.py` -- add to `_register_builtin_providers()` (`factory.py:189-217`) with `register_provider("gemini", GeminiProvider)`.
3. Add config section in `kosmos/config.py` -- create a `GeminiConfig(BaseSettings)` class following the `ClaudeConfig` pattern (`config.py:26-84`). Add the section to `KosmosConfig` (`config.py:959-976`).
4. Wire optional config creation following `_optional_claude_config()` pattern (`config.py:914-919`).
5. Handle in `get_provider_from_config()` at `kosmos/core/llm.py:544-607` with a new `elif config.llm_provider == "gemini"` branch.

### Adding a New CLI Command

To add a new CLI command, start at `kosmos/cli/main.py:397-415` for the registration pattern.

1. Create `kosmos/cli/commands/your_command.py` -- define a function decorated with click/typer conventions (follow `kosmos/cli/commands/run.py` pattern).
2. Add the import to `register_commands()` in `kosmos/cli/main.py:401-410`. The import name must match the module name.
3. Register with `app.command()` in the same function.

### Adding a New Database Model

To add a new ORM model, start at `kosmos/db/models.py` for existing model patterns.

1. Add SQLAlchemy model class in `kosmos/db/models.py` -- inherit from `Base` (`models.py:19`). Follow the `Experiment` model pattern (`models.py:36-72`).
2. Add CRUD functions in `kosmos/db/operations.py` -- follow the `create_hypothesis()`/`get_hypothesis()` pattern. Use `get_session()` context manager for all DB ops.
3. Create Alembic migration: `alembic revision --autogenerate -m "add_your_model"`. Existing migrations are in `alembic/versions/`.
4. Add Pydantic counterpart in `kosmos/models/` for API/serialization use (keep ORM and Pydantic models separate).

### Adding a New Event Type

To add a new event for the pub/sub system, start at `kosmos/core/events.py`.

1. Add the event type to the `EventType` enum (`events.py:15-52`). Currently has 16 types across 5 categories.
2. Create a new event dataclass if needed, following `StreamingEvent` pattern (`events.py`).
3. Publish from your code via `get_event_bus().publish_sync(event)` or `await get_event_bus().publish(event)` (`event_bus.py:28-56`).
4. Subscribe in consuming code via `get_event_bus().subscribe(callback, event_types=[EventType.YOUR_TYPE])` (`event_bus.py:58-84`).
5. If the event should reach WebSocket clients, update `kosmos/api/websocket.py` to forward it.

### Adding a New Experiment Template

To add a template for an existing domain, start at `kosmos/experiments/templates/base.py:98-336` for the `TemplateBase` ABC.

1. Create your template file in the appropriate domain directory (e.g., `kosmos/experiments/templates/biology/your_template.py`).
2. Subclass `TemplateBase` and call `super().__init__()` with `name`, `experiment_type`, `domain`, `title`, and `description` (min 50 chars) (`base.py:128-147`).
3. Implement two abstract methods: `generate_protocol()` (`base.py:149-166`) and `is_applicable()` (`base.py:168-179`).
4. Import in the domain's `templates/__init__.py`. Auto-discovery handles registration if the domain package is in `template_packages` (`base.py:392-396`).

### Adding a New Monitoring Alert Rule

To add a custom alert rule, start at `kosmos/monitoring/alerts.py`.

1. Define a rule function following the pattern of existing rules (`alerts.py:240-321`). Accepts a metrics dict, returns True if the alert condition is met.
2. Register the rule in the `DEFAULT_RULES` list or via `AlertManager.add_rule()`.
3. Note: 3 of the 7 existing default rules are placeholders that always return False (`alerts.py:284-321`). You may want to implement these first.

### Adding a Config Section

To add a new configuration section, start at `kosmos/config.py`.

1. Create a new `YourConfig(BaseSettings)` class with Pydantic fields and validators (follow `ResearchConfig` at `config.py:193-250` as a pattern).
2. Add the field to `KosmosConfig` (`config.py:959-976`) as `your_config: YourConfig = YourConfig()`.
3. If the config reads env vars, be aware of the nested BaseSettings inheritance issue -- env vars may not propagate automatically. See the `sync_litellm_env_vars` workaround at `config.py:986-1022`.
4. Add entries to `create_directories()` if filesystem paths are involved (`config.py:1067-1076`).

### Adding a World Model Backend

To add a new storage backend for the world model, start at `kosmos/world_model/interface.py` for the `WorldModel` protocol.

1. Create `kosmos/world_model/your_backend.py` -- implement the `WorldModel` interface.
2. Update the factory in `kosmos/world_model/factory.py` -- add a new branch in `get_world_model()` (`factory.py:105-155`). Currently supports Neo4j with InMemory fallback.
3. Provide a `reset_world_model()` implementation for test isolation (`factory.py:181`).
## Change Playbooks

### Playbook 1: Add a New Agent Type with Custom Behavior

This playbook covers adding a new agent to the Kosmos research pipeline -- for example, a `PeerReviewerAgent` that validates experimental results against published literature before convergence. The system currently has 6 agent files [FACT: kosmos/agents/ directory contains base.py, data_analyst.py, experiment_designer.py, hypothesis_generator.py, literature_analyzer.py, research_director.py] plus a registry [FACT: kosmos/agents/registry.py] and skill_loader [FACT: kosmos/agents/skill_loader.py]. Agents communicate via `AgentMessage` objects [FACT: kosmos/agents/base.py:45-85] using async message passing with sync wrappers [FACT: kosmos/agents/base.py:302-327]. The `AgentRegistry` [FACT: kosmos/agents/registry.py:24-526] manages discovery, routing, and lifecycle. The `ResearchDirectorAgent` [FACT: kosmos/agents/research_director.py:53-66] orchestrates all agents through both message-based and direct-call patterns [FACT: research_director module findings].

#### Step 1: Create the Agent Module File

**File to create:** `kosmos/agents/peer_reviewer.py`

Follow the constructor pattern used by all existing agents. Every agent constructor calls `super().__init__()` with three positional args: `agent_id`, `agent_type`, and `config` [FACT: kosmos/agents/base.py:113-129]. The `agent_type` defaults to the class name if not provided [FACT: kosmos/agents/base.py:128].

Import the required base types:
```python
from kosmos.agents.base import BaseAgent, AgentMessage, MessageType, AgentStatus
```

The constructor must:
1. Call `super().__init__(agent_id, agent_type or "PeerReviewerAgent", config)` [FACT: pattern from experiment_designer.py:94, hypothesis_generator.py:76, literature_analyzer.py:95]
2. Extract config values with `self.config.get(key, default)` [FACT: experiment_designer.py:97-101, hypothesis_generator.py:79-83]
3. Initialize LLM client via `self.llm_client = get_client()` [FACT: experiment_designer.py:104, hypothesis_generator.py:87, literature_analyzer.py:107]
4. Set up any domain-specific components

**Key behavioral contract:** `BaseAgent.__init__` sets `self.status = AgentStatus.CREATED` [FACT: kosmos/agents/base.py:131] and initializes both a legacy sync message queue (`self.message_queue: List[AgentMessage]`) and an async queue (`self._async_message_queue: asyncio.Queue`) [FACT: kosmos/agents/base.py:136-137]. Your subclass must not re-initialize these.

#### Step 2: Implement Required Methods

**2a: `execute()` method**

This is the primary task entry point. The base class declares `execute(self, task: Dict[str, Any]) -> Dict[str, Any]` and raises `NotImplementedError` [FACT: kosmos/agents/base.py:485-497]. Existing agents use two different signatures:

- `ExperimentDesignerAgent.execute(self, message: AgentMessage) -> AgentMessage` [FACT: experiment_designer.py:109-160]
- `LiteratureAnalyzerAgent.execute(self, task: Dict[str, Any]) -> Dict[str, Any]` [FACT: literature_analyzer.py:153-229]

Choose whichever matches your use case; the ResearchDirector handles both patterns via its dual communication system [FACT: research_director module findings].

Pattern for `execute()`:
1. Set `self.status = AgentStatus.WORKING` [FACT: experiment_designer.py:119, literature_analyzer.py:180]
2. Dispatch on `task_type` or `message.content.get("task_type")` [FACT: experiment_designer.py:122, literature_analyzer.py:183]
3. Increment `self.tasks_completed` on success [FACT: literature_analyzer.py:221]
4. Set `self.status = AgentStatus.IDLE` on completion [FACT: experiment_designer.py:163, literature_analyzer.py:222]
5. Set `self.status = AgentStatus.ERROR` on failure and increment `self.errors_encountered` [FACT: experiment_designer.py:156, literature_analyzer.py:226]

**2b: `process_message()` method (optional but recommended)**

Override the async `process_message(self, message: AgentMessage)` from BaseAgent [FACT: kosmos/agents/base.py:382-392] to handle incoming inter-agent messages. The base implementation only logs a warning [FACT: kosmos/agents/base.py:391]. When a message arrives via `receive_message()`, it is automatically placed on both queues and then `process_message()` is called [FACT: kosmos/agents/base.py:354-359].

**2c: Lifecycle hooks (optional)**

Override `_on_start()` and `_on_stop()` for initialization/cleanup [FACT: kosmos/agents/base.py:503-509]. The LiteratureAnalyzerAgent uses `_on_start()` to record start time and `_on_stop()` to log runtime statistics [FACT: literature_analyzer.py:138-152].

#### Step 3: Add Domain-Specific Logic

Create the agent's core analytical methods. Follow the pattern of separating public API methods from internal helpers:
- Public: `design_experiment()`, `generate_hypotheses()`, `summarize_paper()` [FACT: experiment_designer.py:164-299, hypothesis_generator.py:142-257, literature_analyzer.py:231-291]
- Private helpers: `_load_hypothesis()`, `_validate_protocol()`, `_build_summarization_prompt()` [FACT: experiment_designer.py:353-378, 720-777, literature_analyzer.py:870-896]

Each public method should:
1. Log entry with `logger.info()` [FACT: experiment_designer.py:192, hypothesis_generator.py:173]
2. Perform multi-step processing with clear stages [FACT: experiment_designer.py:195-292 has 8 labeled steps]
3. Optionally store results to DB via `get_session()` context manager [FACT: experiment_designer.py:885-922, hypothesis_generator.py:463-500]
4. Return a Pydantic model or structured dict [FACT: experiment_designer.py:277-293]

#### Step 4: Register in `kosmos/agents/__init__.py`

**File to modify:** `kosmos/agents/__init__.py`

The `__init__.py` explicitly imports and re-exports all agent classes [FACT: kosmos/agents/__init__.py:1-29]. Add your import and `__all__` entry:
```python
from .peer_reviewer import PeerReviewerAgent
```
Add `"PeerReviewerAgent"` to the `__all__` list [FACT: kosmos/agents/__init__.py:22-29].

#### Step 5: Wire into the AgentRegistry

**File to understand:** `kosmos/agents/registry.py`

The `AgentRegistry` [FACT: kosmos/agents/registry.py:24-526] uses a singleton pattern via `get_registry()` [FACT: kosmos/agents/registry.py:512-526]. Registration happens when the caller calls `registry.register(agent_instance)` [FACT: kosmos/agents/registry.py:70-97]. This automatically sets up message routing by calling `agent.set_message_router(self._route_message)` [FACT: kosmos/agents/registry.py:94].

Your agent does not need code changes here -- registration is caller-side. The singleton getter `get_registry()` [FACT: kosmos/agents/registry.py:512-526] provides access to the shared instance. But you must ensure your agent is instantiated and registered in the code that sets up the research pipeline (typically the ResearchDirector or CLI entry point). The registry tracks agents by both ID and type [FACT: kosmos/agents/registry.py:59-60] and provides health monitoring via `get_system_health()` [FACT: kosmos/agents/registry.py:436-466].

#### Step 6: Wire into the ResearchDirector (if participating in research loop)

**File to modify:** `kosmos/agents/research_director.py`

The ResearchDirector uses a lazy-init pattern for sub-agents [FACT: research_director.py:145-152]. Each agent slot is initialized to `None` and instantiated on first use [FACT: research_director module findings]. To add your agent:

1. Add a slot: `self._peer_reviewer = None` in `__init__` alongside the other agent slots [FACT: research_director.py:145-152]
2. Add to `self.agent_registry` dict mapping: `agent_type -> agent_id` [FACT: research_director.py:142]
3. Add a `_handle_peer_review_action()` method following the existing `_handle_*_action()` pattern [FACT: research_director module findings: handler methods at lines 1391-1979]
4. Add a `_send_to_peer_reviewer()` async method following the `_send_to_*` pattern [FACT: research_director module findings: send methods at lines 1039-1219]
5. Update `decide_next_action()` [FACT: research_director.py:~2388-2548] to include your agent in the workflow state machine if it should participate in the research loop

**Critical:** The ResearchDirector supports both message-based and direct-call patterns [FACT: research_director module findings]. If your agent is a utility (like ConvergenceDetector), use direct calls. If it participates in the async message flow, use message-based coordination. (see Gotcha #1 -- silent message routing failure for unregistered agents; see Gotcha #12 -- dual communication maintenance burden)

#### Step 7: Add Pydantic Models (if needed)

**Directory:** `kosmos/models/`

If your agent produces structured output, create Pydantic models following the existing pattern. See `kosmos/models/hypothesis.py` [FACT: hypothesis_generator.py imports Hypothesis, HypothesisGenerationRequest, HypothesisGenerationResponse] and `kosmos/models/experiment.py` [FACT: experiment_designer.py imports ExperimentProtocol, ExperimentDesignRequest, ExperimentDesignResponse, etc.]. The LiteratureAnalyzer uses a plain dataclass `PaperAnalysis` instead of Pydantic [FACT: literature_analyzer.py:33-47], showing that simpler result types can use `@dataclass` with an explicit `to_dict()` method [FACT: literature_analyzer.py:45-46].

If your model has enum fields, decide on `ConfigDict(use_enum_values=False)` (Hypothesis pattern -- enums as objects, manual `.value` in `to_dict()`) [FACT: hypothesis.py:156] vs `ConfigDict(use_enum_values=True)` (WorkflowTransition pattern -- enums as strings automatically) [FACT: workflow.py:48]. The codebase is inconsistent here -- pick one and document it. (see Gotcha #71 -- `to_dict()` vs `model_dump()` divergence)

#### Step 8: Add Database Models (if persisting data)

**Directory:** `kosmos/db/models.py`

If your agent persists data, add SQLAlchemy models following existing patterns. The experiment_designer stores protocols via `DBExperiment` [FACT: experiment_designer.py:39-42], and the hypothesis_generator stores via `DBHypothesis` [FACT: hypothesis_generator.py:27]. Use `get_session()` context manager for all DB operations [FACT: experiment_designer.py:888, hypothesis_generator.py:474]. Note the dual model system: you will need BOTH a Pydantic model (runtime) and an SQLAlchemy model (persistence) with manual conversion between them [FACT: hypothesis module findings: "Dual model system"].

Add corresponding CRUD functions to `kosmos/db/operations.py` following the pattern: validate JSON fields via `_validate_json_dict()`/`_validate_json_list()` [FACT: operations.py:27-45], then `session.add()`, `session.commit()`, `session.refresh()` [FACT: operations.py:113-114]. Each CRUD function commits immediately [FACT: operations module findings: "Immediate commits"] (see Gotcha #65 -- immediate commits prevent atomic batching).

#### Step 9: Add SkillLoader Integration (optional)

**File to modify:** `kosmos/agents/skill_loader.py`

If your agent needs domain-specific prompt injection, add a skill bundle to `SKILL_BUNDLES` dict [FACT: skill_loader.py:46-82] and a domain mapping to `DOMAIN_TO_BUNDLES` dict [FACT: skill_loader.py:91-102]. Call `skill_loader.load_skills_for_task()` in your agent's methods [FACT: skill_loader.py:328-385]. The formatted output is a markdown string suitable for LLM prompt injection [FACT: skill_loader.py:387-423]. Missing skills are handled gracefully [FACT: skill_loader.py:239-241]. The `_format_skills_for_prompt()` method limits output to 15 skills to avoid token bloat [FACT: skill_loader.py:408].

#### Step 10: Write Tests

**File to create:** `tests/unit/agents/test_peer_reviewer.py`

Follow the test pattern from existing agent tests [FACT: tests/unit/agents/test_hypothesis_generator.py:1-80]:

1. Use `pytest.mark.requires_claude` and skip if API key unavailable [FACT: test_hypothesis_generator.py:20-26]
2. Create a fixture that instantiates the agent with test config [FACT: test_hypothesis_generator.py:34-41]
3. Test initialization with defaults and custom config [FACT: test_hypothesis_generator.py:48-67]
4. Test `execute()` with mock messages
5. Test core domain methods with real or mocked LLM calls
6. Test error handling and status transitions

Minimum test coverage:
- `test_init_default` -- verify default config values [FACT: test_hypothesis_generator.py:48-55]
- `test_init_with_config` -- verify custom config overrides [FACT: test_hypothesis_generator.py:57-67]
- `test_execute_success` -- happy path with mock message
- `test_execute_unknown_task` -- should raise ValueError [FACT: experiment_designer.py:148]
- `test_execute_error_handling` -- should set status to ERROR [FACT: experiment_designer.py:156]
- `test_lifecycle` -- start/stop/status transitions
- `test_message_handling` -- send/receive messages

#### Validation Commands

```bash
# Verify import works
python -c "from kosmos.agents.peer_reviewer import PeerReviewerAgent; print('OK')"

# Verify __init__ exports
python -c "from kosmos.agents import PeerReviewerAgent; print('OK')"

# Verify agent instantiation
python -c "
from kosmos.agents.peer_reviewer import PeerReviewerAgent
agent = PeerReviewerAgent(config={})
print(f'Type: {agent.agent_type}, Status: {agent.status}')
"

# Verify registry integration
python -c "
from kosmos.agents.registry import get_registry
from kosmos.agents.peer_reviewer import PeerReviewerAgent
reg = get_registry(reset=True)
agent = PeerReviewerAgent()
reg.register(agent)
print(f'Registered: {len(reg)} agents')
print(f'Types: {reg.list_agent_types()}')
"

# Run unit tests
python -m pytest tests/unit/agents/test_peer_reviewer.py -v

# Verify no broken imports
python -c "import kosmos.agents; print('All agent imports OK')"
```

#### Common Mistakes

**Mistake 1: Forgetting to call `super().__init__()` correctly.** The BaseAgent constructor initializes essential state: `status`, `created_at`, `message_queue`, `_async_message_queue`, and counters (`messages_received`, `messages_sent`, etc.) [FACT: kosmos/agents/base.py:127-151]. Skipping `super().__init__()` or passing wrong args causes AttributeError at runtime. Always call `super().__init__(agent_id, agent_type or "YourAgentName", config)` as the first line [FACT: experiment_designer.py:94, hypothesis_generator.py:76].

**Mistake 2: Making `execute()` async when using direct-call pattern.** The ResearchDirector has two communication modes [FACT: research_director module findings]. The `_handle_*_action()` methods call agents directly (synchronous), while `_send_to_*` methods use async message passing. If your agent is called via direct pattern, its `execute()` must be synchronous (return a value, not a coroutine). The base class `execute()` is sync [FACT: kosmos/agents/base.py:485-497]. Only `process_message()`, `send_message()`, and `receive_message()` are async [FACT: kosmos/agents/base.py:246-404].

**Mistake 3: Not setting status transitions correctly.** Agent status must follow the lifecycle: CREATED -> STARTING -> RUNNING -> WORKING -> IDLE -> STOPPED [FACT: kosmos/agents/base.py:25-34]. The `is_healthy()` check returns True only for RUNNING, IDLE, and WORKING [FACT: kosmos/agents/base.py:217]. If you forget to set status back to IDLE after task completion, the agent's health check will fail and the registry will report it as unhealthy [FACT: kosmos/agents/registry.py:436-466].

**Mistake 4: Not handling message routing for new agent type.** When registered in `AgentRegistry`, the agent's `set_message_router()` is called with the registry's `_route_message` callback [FACT: kosmos/agents/registry.py:94]. But `send_message()` only works if the target agent is also registered [FACT: kosmos/agents/registry.py:240-243]. If you send a message to an unregistered agent, the error is logged but silently swallowed [FACT: kosmos/agents/registry.py:242-243]. This caused the ConvergenceDetector issue that led to the dual communication pattern [FACT: research_director module findings].

**Mistake 5: Forgetting to add to `__all__` in `__init__.py`.** The `kosmos/agents/__init__.py` has both explicit imports and an `__all__` list [FACT: kosmos/agents/__init__.py:15-29]. Adding the import but forgetting the `__all__` entry means `from kosmos.agents import *` will not include your agent. Conversely, adding to `__all__` without the import causes `ImportError`.

**Mistake 6: Using blocking I/O in async `process_message()`.** The `process_message()` method is async [FACT: kosmos/agents/base.py:382]. If you call synchronous blocking functions (e.g., `time.sleep()`, blocking HTTP requests) inside it, you block the event loop. This is a known issue in the codebase -- the ResearchDirector itself has `time.sleep()` in an async error handler [FACT: research_director module findings]. Use `await asyncio.sleep()` and async HTTP clients.

**Mistake 7: Not implementing error response for REQUEST messages.** When `receive_message()` catches an exception from `process_message()`, it automatically sends an ERROR response back to the sender IF the incoming message type was REQUEST [FACT: kosmos/agents/base.py:361-366]. Your `process_message()` should raise exceptions for genuine errors rather than silently returning, so the caller gets notified. But also be aware this means any unhandled exception triggers an outbound message.

**Mistake 8: Storing agent state without using `save_state_data()`.** The BaseAgent provides `save_state_data(key, value)` and `get_state_data(key, default)` for state persistence [FACT: kosmos/agents/base.py:472-479]. Using `self.state_data[key] = value` directly skips the `updated_at` timestamp update [FACT: kosmos/agents/base.py:475]. Always use the accessor methods to ensure state timestamps are correct for serialization via `get_state()` [FACT: kosmos/agents/base.py:439-454].

**Mistake 9: Creating a singleton factory without a reset mechanism.** Both `LiteratureAnalyzerAgent` and `AgentRegistry` provide singleton getters with `reset` parameters [FACT: literature_analyzer.py:1053-1076, kosmos/agents/registry.py:512-526]. If your agent uses a singleton pattern, always provide a `reset=False` parameter for testing. Without it, test isolation fails because agent state leaks between tests.

**Mistake 10: Not adding the agent to the ResearchDirector's lazy-init slots.** If your agent participates in the research workflow, it must be added as a lazy-init slot (`self._peer_reviewer = None`) [FACT: research_director.py:145-152] AND instantiated on first use. Instantiating in `__init__` causes unnecessary overhead and potential circular imports. The lazy pattern avoids importing the agent module until actually needed [FACT: research_director module findings].

**Mistake 11: Choosing the wrong serialization strategy for Pydantic models.** The codebase has three serialization strategies (manual `to_dict()` with `.value` calls, `model_to_dict()` compat wrapper, Pydantic default). If your agent produces results that cross module boundaries or get persisted, use manual `to_dict()` with explicit enum `.value` conversion matching the hypothesis/experiment pattern [FACT: hypothesis.py:117-142, experiment.py:471-573]. If you use Pydantic's default `model_dump()` with `use_enum_values=False` [FACT: hypothesis.py:156], consumers will receive enum objects instead of strings.

---

### Playbook 2: Add a New Scientific Domain

This playbook covers adding a fully integrated scientific domain to Kosmos -- using "ecology" as the example. The system currently has 5 domain directories: biology [FACT: kosmos/domains/biology/__init__.py:1-76], chemistry (empty stub) [FACT: kosmos/domains/chemistry/__init__.py is 0 bytes], materials [FACT: kosmos/domains/materials/__init__.py:1-59], neuroscience [FACT: kosmos/domains/neuroscience/__init__.py:1-72], and physics (empty stub) [FACT: kosmos/domains/physics/__init__.py is 0 bytes]. Each fully-implemented domain has experiment templates under `kosmos/experiments/templates/<domain>/` [FACT: templates/biology/ has 2 templates, templates/neuroscience/ has 2, templates/materials/ has 3] and skill bundle mappings in `SkillLoader` [FACT: kosmos/agents/skill_loader.py:91-102]. There are 13 distinct registration points that must be updated for full integration.

#### Step 1: Create the Domain Directory Structure

**Directory to create:** `kosmos/domains/ecology/`

Fully-implemented domains (biology, neuroscience, materials) follow a four-file pattern [FACT: biology/ has 4 .py files, neuroscience/ has 4 .py files, materials/ has 4 .py files]:

| File | Purpose | Example Reference |
|------|---------|-------------------|
| `__init__.py` | Re-exports all public classes | biology: 76 lines [FACT: kosmos/domains/biology/__init__.py] |
| `apis.py` | HTTP clients for external databases | biology: 28149 bytes, 10 clients [FACT: biology/apis.py:1-15] |
| `ontology.py` | Domain knowledge graph with concepts + relations | biology: 15550 bytes [FACT: biology/ontology.py:1-80] |
| `<analyzer>.py` | Domain-specific analysis algorithms | biology/genomics.py: 19638 bytes [FACT: biology/genomics.py:1-100] |

Chemistry and physics are stub domains with only empty `__init__.py` files [FACT: chemistry/__init__.py is 0 bytes, physics/__init__.py is 0 bytes]. Biology is the most complete at 4 files totaling approximately 81KB [FACT: biology/ contains apis.py at 28149 bytes, genomics.py at 19638 bytes, metabolomics.py at 17739 bytes, ontology.py at 15550 bytes].

#### Step 2: Create API Clients (`apis.py`)

**File to create:** `kosmos/domains/ecology/apis.py`

Follow the biology/apis.py pattern [FACT: kosmos/domains/biology/apis.py:1-15]:

1. Define dataclass models for API response data [FACT: biology/apis.py:27-57 defines KEGGPathway, GWASVariant, eQTLData as dataclasses]
2. Create client classes for each external database. Biology has 10 clients [FACT: biology/__init__.py:3-14], materials has 5 [FACT: materials/__init__.py:3-11]
3. Use `httpx.Client` for HTTP requests [FACT: biology/apis.py:71 uses httpx.Client]
4. Apply `tenacity.retry` decorators with `stop_after_attempt(3)` and `wait_exponential(min=1, max=10)` [FACT: biology/apis.py:73, 97]
5. Each client class has a `BASE_URL` constant and `__init__(self, timeout: int = 30)` [FACT: biology/apis.py:66-71]
6. Return `Optional[Dict]` or `Optional[DataClass]` from methods -- never raise on API failure, return None [FACT: biology/apis.py:93-95]

#### Step 3: Create the Domain Ontology (`ontology.py`)

**File to create:** `kosmos/domains/ecology/ontology.py`

Three patterns exist for ontology implementation:

- **Pattern A (Biology-native):** Defines own `BiologicalRelationType`, `BiologicalConcept`, `BiologicalRelation` as Pydantic models [FACT: kosmos/domains/biology/ontology.py:31-64]
- **Pattern B (Neuroscience reuse):** Imports from biology: `from kosmos.domains.biology.ontology import BiologicalRelationType, BiologicalConcept, BiologicalRelation` [FACT: kosmos/domains/neuroscience/ontology.py:29-34]
- **Pattern C (Materials standalone):** Defines own `MaterialsRelationType`, `MaterialsConcept`, `MaterialsRelation` [FACT: kosmos/domains/materials/ontology.py:32-60]

If ecology shares concepts with biology, use Pattern B. If it needs fundamentally different relation types, use Pattern C. Biology has 9 relation types [FACT: biology/ontology.py:33-41] and materials has 9 [FACT: materials/ontology.py:34-43]. All use the `str, Enum` pattern [FACT: biology/ontology.py:31, materials/ontology.py:32].

Each ontology class initializes `self.concepts: Dict[str, Concept]` and `self.relations: List[Relation]` [FACT: biology/ontology.py:78-79, neuroscience/ontology.py:51-52, materials/ontology.py:~68], with private `_initialize_*()` methods that populate the knowledge graph. Neuroscience has 5 initializers [FACT: neuroscience/ontology.py:55-59]. Public query methods: `get_concept()`, `get_child_concepts()`, `find_related_concepts()` [FACT: neuroscience/ontology.py docstring:15-23].

#### Step 4: Create Domain Analyzers

**File to create:** `kosmos/domains/ecology/<analyzer_name>.py`

Each domain has 1-3 analyzer classes:
- Biology: `GenomicsAnalyzer` [FACT: biology/genomics.py:1-100] and `MetabolomicsAnalyzer` [FACT: biology/__init__.py:16-23]
- Neuroscience: `ConnectomicsAnalyzer` [FACT: neuroscience/__init__.py:17-22] and `NeurodegenerationAnalyzer` [FACT: neuroscience/__init__.py:25-32]
- Materials: `MaterialsOptimizer` [FACT: materials/__init__.py:16-22]

Analyzers produce Pydantic result models [FACT: biology/genomics.py:68-100 defines CompositeScore with Field validators] and use domain API clients for data [FACT: biology/genomics.py:40-46 imports from apis]. Use `numpy`, `pandas`, and `scipy.stats` as biology/genomics does [FACT: biology/genomics.py:35-37].

#### Step 5: Wire the Domain `__init__.py`

**File to create:** `kosmos/domains/ecology/__init__.py`

Re-export all public classes with `__all__` [FACT: biology/__init__.py:41-75]. The module docstring should name the domain [FACT: biology/__init__.py:1, neuroscience/__init__.py:1, materials/__init__.py:1]. Neuroscience exports 14 classes [FACT: neuroscience/__init__.py:38-71] and materials exports 14 classes [FACT: materials/__init__.py:31-58].

#### Step 6: Create Experiment Templates

**Directory to create:** `kosmos/experiments/templates/ecology/`

Each template extends `TemplateBase` [FACT: kosmos/experiments/templates/base.py:98-336] and must implement two abstract methods:
1. `generate_protocol(self, params: TemplateCustomizationParams) -> ExperimentProtocol` [FACT: base.py:149-166]
2. `is_applicable(self, hypothesis: Hypothesis) -> bool` [FACT: base.py:168-179]

The constructor calls `super().__init__()` with: `name`, `experiment_type` (ExperimentType enum: COMPUTATIONAL, DATA_ANALYSIS, LITERATURE_SYNTHESIS [FACT: base.py:367-371]), `domain` (must match your domain string), `title`, `description` (min 50 chars [FACT: base.py:37]) [FACT: base.py:128-147].

Create `kosmos/experiments/templates/ecology/__init__.py` following [FACT: kosmos/experiments/templates/biology/__init__.py:1-13].

**Register in template auto-discovery.** Modify `kosmos/experiments/templates/base.py` at lines 392-396 [FACT: base.py:392-396] to add `'kosmos.experiments.templates.ecology'` to the `template_packages` list. Auto-discovery uses `pkgutil.iter_modules()` [FACT: base.py:407] and `inspect.getmembers(module, inspect.isclass)` [FACT: base.py:417-420] to find all `TemplateBase` subclasses. Failed instantiations are silently caught [FACT: base.py:426-429].

#### Step 7: Register in Configuration and CLI (THREE separate lists)

**7a: `kosmos/config.py:210-213`** -- Add `"ecology"` to `enabled_domains` default list [FACT: config.py:210-213]. The field uses `parse_comma_separated` via `BeforeValidator` [FACT: config.py:210].

**7b: `kosmos/cli/commands/config.py:246`** -- Add `"ecology"` to `valid_domains` set [FACT: cli/commands/config.py:246]. Without this, `kosmos config check` reports the domain as invalid.

**7c: `kosmos/agents/research_director.py:265`** -- Add `"ecology"` to `default_domains` list [FACT: research_director.py:265]. Without this, the director logs: `"Domain 'ecology' not in enabled domains"` [FACT: research_director.py:269-272] and says "Research will proceed but domain-specific features may be limited" [FACT: research_director.py:271-272].

#### Step 8: Register in SkillLoader

**File to modify:** `kosmos/agents/skill_loader.py`

Add an ecology skill bundle to `SKILL_BUNDLES` dict [FACT: skill_loader.py:46-82] and an ecology domain mapping to `DOMAIN_TO_BUNDLES` dict [FACT: skill_loader.py:91-102]. Missing skill files are handled gracefully -- the SkillLoader logs warnings but does not crash [FACT: skill_loader.py:239-241]. The `_format_skills_for_prompt()` method limits output to 15 skills [FACT: skill_loader.py:408].

#### Step 9: Add to ExperimentDesigner Domain Defaults

**File to modify:** `kosmos/agents/experiment_designer.py`

Add your domain to the `domain_defaults` dict [FACT: experiment_designer.py:394-404] which maps domains to preferred experiment types. The current dict has 7 entries [FACT: experiment_designer.py:395-403]. Without this, the designer falls through to `ExperimentType.COMPUTATIONAL` as the default [FACT: experiment_designer.py:404].

#### Step 10: Write Tests

Create test files:
- `tests/unit/domains/test_ecology_apis.py` -- API client tests with mocked HTTP [FACT: tests/unit/agents/test_hypothesis_generator.py:20-26 pattern]
- `tests/unit/domains/test_ecology_ontology.py` -- Verify concepts initialized, relations bidirectional [FACT: neuroscience/ontology.py:55-59 pattern]
- `tests/unit/experiments/test_ecology_templates.py` -- Template instantiation, `is_applicable()`, `generate_protocol()` produces valid `ExperimentProtocol` [FACT: base.py:149-166], `validate_template()` passes [FACT: base.py:181-204]

Integration test:
```python
from kosmos.experiments.templates.base import get_template_registry
registry = get_template_registry()
ecology_templates = registry.get_templates_by_domain("ecology")
assert len(ecology_templates) > 0
```

#### Validation Commands

```bash
# Verify domain module imports
python -c "from kosmos.domains.ecology import EcologyOntology; print('OK')"

# Verify ontology initializes
python -c "
from kosmos.domains.ecology.ontology import EcologyOntology
ont = EcologyOntology()
print(f'Concepts: {len(ont.concepts)}, Relations: {len(ont.relations)}')
"

# Verify template auto-discovery
python -c "
from kosmos.experiments.templates.base import get_template_registry
registry = get_template_registry()
ecology = registry.get_templates_by_domain('ecology')
print(f'Ecology templates: {len(ecology)}')
"

# Verify domain in config defaults
python -c "
from kosmos.config import get_config
config = get_config()
print(f'Enabled domains: {config.research.enabled_domains}')
assert 'ecology' in config.research.enabled_domains
"

# Verify CLI validation accepts new domain
python -c "
valid_domains = {'biology', 'neuroscience', 'materials', 'physics', 'chemistry', 'general', 'ecology'}
assert 'ecology' in valid_domains
print('CLI validation: OK')
"

# Verify skill loader mapping
python -c "
from kosmos.agents.skill_loader import SkillLoader
loader = SkillLoader(auto_discover=False)
assert 'ecology' in loader.DOMAIN_TO_BUNDLES
print(f'Ecology bundles: {loader.DOMAIN_TO_BUNDLES[\"ecology\"]}')
"

# Verify experiment designer domain default
python -c "
from kosmos.agents.experiment_designer import ExperimentDesignerAgent
agent = ExperimentDesignerAgent(config={})
print('ExperimentDesigner: OK')
"

# Run all domain tests
python -m pytest tests/unit/domains/test_ecology*.py -v

# Run template tests
python -m pytest tests/unit/experiments/test_ecology*.py -v

# Grep to confirm all registration points
grep -r "ecology" kosmos/config.py kosmos/cli/commands/config.py kosmos/agents/skill_loader.py kosmos/agents/research_director.py kosmos/agents/experiment_designer.py kosmos/experiments/templates/base.py
```

#### Complete Registration Point Checklist

Thirteen distinct registration points must be updated [FACT: playbook findings]. Missing any one causes partial integration with subtle runtime degradation.

| # | File | What to Change | Line Reference |
|---|------|---------------|----------------|
| 1 | `kosmos/domains/ecology/__init__.py` | Create with all re-exports | Pattern: biology/__init__.py [FACT: biology/__init__.py:1-76] |
| 2 | `kosmos/domains/ecology/apis.py` | API clients with retry | Pattern: biology/apis.py [FACT: biology/apis.py:1-96] |
| 3 | `kosmos/domains/ecology/ontology.py` | Knowledge graph | Pattern: biology/ontology.py [FACT: biology/ontology.py:1-80] |
| 4 | `kosmos/domains/ecology/<analyzer>.py` | Domain analysis | Pattern: biology/genomics.py [FACT: biology/genomics.py:1-100] |
| 5 | `kosmos/experiments/templates/ecology/__init__.py` | Template exports | Pattern: biology templates [FACT: experiments/templates/biology/__init__.py:1-13] |
| 6 | `kosmos/experiments/templates/ecology/<template>.py` | Experiment templates | Pattern: metabolomics_comparison.py [FACT: metabolomics_comparison.py:48-80] |
| 7 | `kosmos/experiments/templates/base.py:392-396` | Add to template_packages list | [FACT: base.py:392-396] |
| 8 | `kosmos/config.py:210-213` | Add to enabled_domains default | [FACT: config.py:210-213] |
| 9 | `kosmos/cli/commands/config.py:246` | Add to valid_domains set | [FACT: cli/commands/config.py:246] |
| 10 | `kosmos/agents/research_director.py:265` | Add to default_domains list | [FACT: research_director.py:265] |
| 11 | `kosmos/agents/skill_loader.py:46-82` | Add SKILL_BUNDLES entry | [FACT: skill_loader.py:46-82] |
| 12 | `kosmos/agents/skill_loader.py:91-102` | Add DOMAIN_TO_BUNDLES entry | [FACT: skill_loader.py:91-102] |
| 13 | `kosmos/agents/experiment_designer.py:394-404` | Add domain_defaults entry | [FACT: experiment_designer.py:394-404] |

#### Common Mistakes

**Mistake 1: Forgetting the template_packages entry in base.py.** The `TemplateRegistry._discover_templates()` uses a hardcoded list of package names [FACT: kosmos/experiments/templates/base.py:392-396]. If you create templates but forget to add `'kosmos.experiments.templates.ecology'` to this list, your templates will not be auto-discovered. The system falls back to LLM-generated protocols instead of using your curated templates [FACT: experiment_designer.py:421-424 falls back to _generate_with_claude()].

**Mistake 2: Missing the CLI valid_domains set.** The `valid_domains` set in `kosmos/cli/commands/config.py:246` is a set literal, not derived from any other source [FACT: config.py:246-247]. If your domain is in `enabled_domains` but not in `valid_domains`, running `kosmos config check` reports the configuration as invalid.

**Mistake 3: Creating an empty `__init__.py` like chemistry/physics.** Chemistry and physics have zero-byte `__init__.py` files [FACT: chemistry/__init__.py is 0 bytes, physics/__init__.py is 0 bytes]. These are stub domains with no functionality. If you create a domain with actual modules but leave `__init__.py` empty, `from kosmos.domains.ecology import EcologyOntology` raises `ImportError`. Always populate `__init__.py` with explicit imports and `__all__` [FACT: biology/__init__.py:1-76].

**Mistake 4: Not matching the TemplateBase abstract methods.** `TemplateBase` is an `ABC` [FACT: base.py:98] requiring `generate_protocol()` [FACT: base.py:149-166] and `is_applicable()` [FACT: base.py:168-179]. If you forget either, auto-discovery silently swallows the `TypeError` in a bare `except Exception` [FACT: base.py:426-429]. Your template simply does not appear in the registry with no error message.

**Mistake 5: Using wrong domain string casing.** Domain strings are compared case-insensitively in some places but case-sensitively in others. `TemplateRegistry.register()` lowercases domain names [FACT: base.py:472], `get_templates_by_domain()` lowercases the query [FACT: base.py:538], `ExperimentDesignerAgent.domain_defaults` uses lowercase keys [FACT: experiment_designer.py:404], `ResearchDirector._validate_domain()` lowercases both sides [FACT: research_director.py:269], `SkillLoader.DOMAIN_TO_BUNDLES` uses lowercase keys [FACT: skill_loader.py:91-102]. Always use lowercase.

**Mistake 6: Not following the retry pattern for API clients.** Biology and materials API clients use `tenacity.retry` with `stop_after_attempt(3)` and `wait_exponential(min=1, max=10)` [FACT: biology/apis.py:73, 97]. If you skip retries, transient network failures cause immediate errors. The catch clause should handle `(httpx.HTTPError, RetryError, Exception)` [FACT: biology/apis.py:93].

**Mistake 7: Importing from biology ontology without dependency awareness.** Neuroscience imports `BiologicalRelationType`, `BiologicalConcept`, `BiologicalRelation` from biology [FACT: neuroscience/ontology.py:29-34]. This creates a cross-domain dependency -- if biology is removed or refactored, neuroscience breaks. Materials avoids this by defining standalone types [FACT: materials/ontology.py:32-43].

**Mistake 8: Forgetting to set `domain` in template metadata.** `TemplateBase.__init__` accepts an optional `domain` parameter [FACT: base.py:134] via `TemplateMetadata` [FACT: base.py:34]. If you omit it, `get_templates_by_domain("ecology")` will not find your template [FACT: base.py:470-474]. Always set `domain="ecology"` explicitly [FACT: metabolomics_comparison.py:63, connectome_scaling.py:60-68].

**Mistake 9: Not handling the TemplateMetadata description minimum length.** `TemplateMetadata.description` has `min_length=50` [FACT: base.py:37]. `validate_template()` also warns if description is under 20 chars [FACT: base.py:197-198]. Shorter descriptions cause Pydantic validation failure during instantiation. Auto-discovery catches this silently [FACT: base.py:426-429], so your template simply does not register.

**Mistake 10: Not updating all three default domain lists simultaneously.** This is the most common integration failure. Three separate lists must be kept in sync: `config.py:211` defaults [FACT: config.py:211], `research_director.py:265` defaults [FACT: research_director.py:265], `cli/commands/config.py:246` valid_domains [FACT: config.py:246]. These are independent literals with no shared source of truth. The codebase already has a discrepancy: `materials` is in `valid_domains` but not in the other two lists [FACT: playbook findings].

**Mistake 11: Not setting `experiment_type` correctly in templates.** The `ExperimentType` enum only has 3 values: `COMPUTATIONAL`, `DATA_ANALYSIS`, `LITERATURE_SYNTHESIS` [FACT: base.py:367-371]. If you use a string that does not match these values, the template constructor fails silently during auto-discovery [FACT: base.py:426-429].

**Mistake 12: Forgetting to handle the case where skill files do not exist.** When adding `SKILL_BUNDLES` and `DOMAIN_TO_BUNDLES` entries for ecology, the actual skill files (markdown files in `kosmos-claude-scientific-skills` directory with 116 skill directories [FACT: skill_loader.py:8]) may not exist yet. The SkillLoader handles missing files gracefully [FACT: skill_loader.py:239-241] -- it logs warnings but does not crash. However, ecology-specific prompt injection will be empty until those files are created. The system degrades to generic prompts.

**Mistake 13: Not testing template auto-discovery end-to-end.** The auto-discovery chain has three silent failure points: (1) package not in `template_packages` list [FACT: base.py:392-396], (2) class not a `TemplateBase` subclass [FACT: base.py:417-420], (3) instantiation failure caught by bare `except` [FACT: base.py:426-429]. The only way to verify is to call `get_template_registry().get_templates_by_domain("ecology")` and check the count is > 0. Always include this integration test.
## Reading Order

### Layer 0: Entry Points and Configuration (read first)

| # | File | Lines | Why Read |
|---|------|-------|----------|
| 1 | `kosmos/cli/main.py` | 440 | CLI entry point. Shows startup sequence: logging setup, database init, command registration. Start here to understand how the application boots. |
| 2 | `kosmos/config.py` | 1160 | Central configuration. 16 Pydantic sub-configs, env var mappings, validation. Skim the field definitions; focus on `get_config()` at line 1136 and `KosmosConfig` at line 922. |
| 3 | `kosmos/cli/commands/run.py` | ~200 | The `run_research()` command. Shows how CLI params become a flat config dict and create a `ResearchDirectorAgent`. The primary user-facing entry point. |

### Layer 1: Core Infrastructure (read second)

| # | File | Lines | Why Read |
|---|------|-------|----------|
| 4 | `kosmos/core/workflow.py` | 416 | State machine with 9 states and explicit `ALLOWED_TRANSITIONS`. Controls the research cycle. Small file, fully readable. |
| 5 | `kosmos/core/events.py` | ~100 | `EventType` enum (16 types) and event dataclasses. Defines the vocabulary for the pub/sub system. |
| 6 | `kosmos/core/event_bus.py` | ~280 | `EventBus` singleton. Publish/subscribe with sync and async support. Thread-safe. |
| 7 | `kosmos/core/logging.py` | ~400 | `setup_logging()`, `JSONFormatter`, `TextFormatter`, `ExperimentLogger`. Sets up the logging infrastructure. |
| 8 | `kosmos/core/providers/base.py` | 484 | `LLMProvider` abstract base class, `LLMResponse`, `UsageStats`, `ProviderAPIError`. The foundation for all LLM interactions. |
| 9 | `kosmos/core/providers/factory.py` | ~220 | Provider registry and `get_provider_from_config()`. How LLM providers are selected and instantiated. |
| 10 | `kosmos/core/llm.py` | 706 | `ClaudeClient` (legacy), `get_client()`, `get_provider()`. The LLM client singleton and auto model selection. |

### Layer 2: Data Layer (read third)

| # | File | Lines | Why Read |
|---|------|-------|----------|
| 11 | `kosmos/db/models.py` | ~80 | SQLAlchemy ORM models: Experiment, Hypothesis, Result, Paper, AgentRecord, ResearchSession. Short and essential. |
| 12 | `kosmos/db/__init__.py` | ~180 | `init_database()`, `get_session()` context manager. Database lifecycle and session management. |
| 13 | `kosmos/db/operations.py` | 600 | CRUD functions for all ORM models. Shows the session-per-operation pattern. |
| 14 | `kosmos/models/hypothesis.py` | ~300 | `Hypothesis` Pydantic model. Core domain object with scoring, thresholds, and serialization. |
| 15 | `kosmos/models/experiment.py` | 699 | `ExperimentProtocol`, `ExperimentStep`, `ResourceRequirements`. Experiment modeling with step validation. |
| 16 | `kosmos/models/result.py` | ~400 | `ExperimentResult`, `VariableResult`, `StatisticalTestResult`. Result modeling with export methods. |
| 17 | `kosmos/models/domain.py` | ~100 | `ScientificDomain` enum and `DomainClassification`. Domain routing model. |

### Layer 3: Agent Framework (read fourth)

| # | File | Lines | Why Read |
|---|------|-------|----------|
| 18 | `kosmos/agents/base.py` | 527 | `BaseAgent` abstract class. Lifecycle (start/stop/pause), message passing (send/receive), state management. The contract all agents must follow. |
| 19 | `kosmos/agents/registry.py` | ~530 | `AgentRegistry` singleton. Agent discovery, message routing, health monitoring. |
| 20 | `kosmos/agents/research_director.py` (key sections only) | 2952 | The orchestrator. Read selectively: `__init__` (68-260), `decide_next_action` (2388-2548), `execute` (2868-2909), `_handle_error_with_recovery` (599-684). |
| 21 | `kosmos/agents/hypothesis_generator.py` | ~500 | Representative specialized agent. Shows how agents use LLM clients, store to DB, and interact with the director. |

### Layer 4: Execution Pipeline (read fifth)

| # | File | Lines | Why Read |
|---|------|-------|----------|
| 22 | `kosmos/safety/code_validator.py` | ~300 | Code safety validation before execution. Pattern matching for dangerous operations. |
| 23 | `kosmos/execution/executor.py` | 1066 | `CodeExecutor`. Sandbox vs exec() fallback, code repair loop, return value extraction. The code execution core. |
| 24 | `kosmos/execution/sandbox.py` | ~400 | `DockerSandbox`. Container management, volume mounting, timeout enforcement. |
| 25 | `kosmos/safety/guardrails.py` | 452 | `SafetyGuardrails`. Emergency stop, resource limits (advisory), signal handling. Note: not wired into the execution pipeline. |
| 26 | `kosmos/execution/code_generator.py` | 995 | LLM-based code generation for experiments. Prompt construction and template selection. |

### Layer 5: Knowledge and Literature (read sixth)

| # | File | Lines | Why Read |
|---|------|-------|----------|
| 27 | `kosmos/literature/base_client.py` | ~300 | `BaseLiteratureClient` abstract class and `PaperMetadata` dataclass. The literature search contract. |
| 28 | `kosmos/literature/arxiv_http_client.py` | ~350 | ArXiv search implementation. Shows the concrete client pattern. |
| 29 | `kosmos/literature/reference_manager.py` | 811 | Cross-source paper management. Deduplication, caching, unified search. |
| 30 | `kosmos/knowledge/graph.py` | 1037 | `KnowledgeGraph` Neo4j interface. Node/relationship CRUD, health check, singleton. |
| 31 | `kosmos/world_model/interface.py` | ~100 | `WorldModel` protocol. Defines the abstract interface for world model backends. |
| 32 | `kosmos/world_model/simple.py` (skim) | 1159 | `Neo4jWorldModel`. Concrete implementation. Many methods with similar patterns; skim rather than deep-read. |

### Layer 6: Domain Modules (read as needed)

| # | File | Lines | Why Read |
|---|------|-------|----------|
| 33 | `kosmos/domains/biology/__init__.py` | 76 | Biology domain public API. Shows the domain module convention (re-exports, `__all__`). |
| 34 | `kosmos/domains/biology/ontology.py` | ~400 | `BiologyOntology`. Hierarchical knowledge graph of biological concepts. Reference for new domain ontologies. |
| 35 | `kosmos/domains/biology/apis.py` | 927 | 10 external API clients (KEGG, GWAS, GTEx, etc.). Reference for adding new API clients. |
| 36 | `kosmos/core/domain_router.py` | ~200 | Routes research questions to domains. Uses LLM-based classification. |

### Layer 7: Orchestration and Monitoring (read last)

| # | File | Lines | Why Read |
|---|------|-------|----------|
| 37 | `kosmos/orchestration/delegation.py` | ~400 | `DelegationManager`. Task routing to agents, batch execution, result classification. |
| 38 | `kosmos/orchestration/plan_reviewer.py` | ~400 | `PlanReviewerAgent`. LLM-based plan review with scoring dimensions. |
| 39 | `kosmos/core/metrics.py` | 935 | `MetricsCollector`. Budget tracking, token counting, performance metrics. |
| 40 | `kosmos/monitoring/alerts.py` | 569 | `AlertManager`. Rule evaluation, email/Slack/PagerDuty handlers. |
| 41 | `kosmos/core/convergence.py` | 711 | `ConvergenceDetector`. Determines when research has converged. |
| 42 | `kosmos/api/websocket.py` | ~200 | WebSocket streaming for real-time client updates. |

### Quick Reference: By Task

- **Understanding the research loop**: Files 1, 3, 4, 20 (decide_next_action section), 37
- **Debugging LLM calls**: Files 8, 9, 10, then the specific provider (`kosmos/core/providers/anthropic.py` or `openai.py` or `litellm_provider.py`)
- **Debugging code execution**: Files 22, 23, 24, 26
- **Adding a new agent**: Files 18, 19, 20 (__init__ and handler sections), 21
- **Adding a new domain**: Files 33, 34, 35, then the checklist in Extension Points
- **Understanding data flow**: Files 11, 12, 13, 14, 15, 16
- **Debugging configuration**: File 2, then `/tmp/deep_crawl/sections/config_surface.md`
## Environment Bootstrap

### Required Services (see Configuration Surface)
- **PostgreSQL** — research state persistence (sessions, experiments, results) (see Module Index: db.py)
- **Neo4j** — knowledge graph storage (see Module Index: knowledge_graph.py)
- **Redis** — caching and pub/sub for real-time events (see Configuration Surface)

### Minimum Environment Variables (see Configuration Surface)
```bash
# LLM Provider (at least one required)
ANTHROPIC_API_KEY=sk-ant-...
# or
OPENAI_API_KEY=sk-...

# Database
NEO4J_PASSWORD=...
NEO4J_DATABASE=kosmos  # defaults to "kosmos"

# Optional
KOSMOS_CONFIG=~/.kosmos/config.yaml
PAGERDUTY_INTEGRATION_KEY=...  # for alerting
SLACK_WEBHOOK_URL=...          # for alerting
```

### Setup Commands
```bash
# Install dependencies
pip install -e .

# Initialize database
python -c "from kosmos.db import init_database; init_database()"

# Verify setup
kosmos doctor

# Run tests
pytest tests/ -x -q
```

---
## Gaps

- **Async patterns coverage**: Only 1 async/sync violation detected by xray (time.sleep in cli.py:227). The 440 async functions may have more subtle boundary issues not yet catalogued.
- **kosmos-reference/ subprojects**: The reference implementations (kosmos-karpathy, kosmos-agentic-data-scientist, kosmos-claude-scientific-writer, kosmos-claude-skills-mcp) were scanned but not deeply investigated. They contain additional patterns and conventions.
- **kosmos-claude-scientific-skills/**: 60+ scientific skill scripts were scanned but individual skill behavior was not traced.
- **Docker/deployment**: Container orchestration (docker-compose.yml, Kubernetes manifests if any) was not investigated.
- **Alembic migrations**: Database schema evolution via alembic/ was noted but migration history was not traced.

---
---

## Meta

| Metric | Value |
|--------|-------|
| Investigation tasks | 46/46 complete |
| Coverage scope | 20 modules deep-read, 5 traces, 10 cross-cutting, 3 convention studies, 3 impact analyses, 2 playbooks |
| Evidence | [FACT] citations verified against source |
| Hub modules covered | 10 (logging, config, research_director, hypothesis, base_client, llm, experiment, workflow, result, db) |
| Change playbooks | 2 (Add new agent, Add new scientific domain) |
| Domain facets | async_service, web_api, cli_tool, scientific_computation |
| Scanner version | X-Ray v3.2 |
