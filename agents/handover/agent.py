from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    from langgraph.graph import StateGraph, START, END
except Exception:
    StateGraph = None
    START = "__start__"
    END = "__end__"

from .config import LLM_LANGGRAPH_ENABLED, ChatConfig, CONVERSATION_POLICY
from .data_access import get_system_by_id, get_realtime_intent_spec, get_realtime_query
from .intent import (
    apply_dictionary_rewrite, detect_intent, detect_previous_user_intent, detect_system_id,
    detect_system_id_with_history, is_confident_intent_hint, is_followup_question,
)
from .llm_client import get_langchain_feature_flags, ollama_generate, _env_flag, _use_langchain
from .models import AgentResult, AgentWorkflowState, resolve_response_route
from .prompts import build_answer_prompt
from .response_builder import (
    build_batch_process_fallback, build_chart_answer, build_graph_answer,
    build_overview_fallback, build_table_answer, remove_repeated_step_sections,
)
from .retrieval import compress_search_result, retrieve_docs
from .rewrite import rewrite_question
from .utils import load_json, normalize_whitespace

class HandoverAgent:
    def __init__(
        self,
        json_path: str,
        persist_dir: str = "./chroma",
        collection_name: str = "handover_agent",
        chat_config: Optional[ChatConfig] = None,
    ) -> None:
        self.payload = load_json(json_path)
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.chat_config = chat_config or ChatConfig()

    def _build_structured_payload(
        self,
        system_id: Optional[str],
        intent: str,
    ) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if system_id and intent in {"overview", "batch_process", "batch_flow", "table_lineage"}:
            system = get_system_by_id(self.payload, system_id)
            if not system:
                return "text", None, None, None
            if intent in {"overview", "batch_process"}:
                return "text", None, None, system.get(intent)
            return "graph", system.get(intent), None, None

        realtime_spec = get_realtime_intent_spec(intent)
        if realtime_spec:
            query_id = str(realtime_spec.get("query_id") or intent)
            query_meta = get_realtime_query(self.payload, query_id)
            if query_meta:
                query_meta = dict(query_meta)
                query_meta.setdefault("query_id", query_id)
                query_meta.setdefault("render_type", realtime_spec.get("render_type"))
                if realtime_spec.get("realtime_mode"):
                    query_meta.setdefault("realtime_mode", realtime_spec.get("realtime_mode"))
            return str(realtime_spec.get("render_type") or "table"), None, query_meta, None

        return "text", None, None, None


    def _build_where_filter(self, system_id: Optional[str], intent: str) -> Optional[Dict[str, Any]]:
        if intent in {"overview", "batch_process", "batch_flow", "table_lineage"} and system_id:
            return {"$and": [{"system_id": system_id}, {"section": intent}]}

        realtime_spec = get_realtime_intent_spec(intent)
        if realtime_spec:
            query_id = str(realtime_spec.get("query_id") or intent)
            return {"$and": [{"section": "realtime_query"}, {"query_id": query_id}]}

        return None


    def answer_question(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 4,
    ) -> AgentResult:
        """질의 응답 진입점.

        LLM_LANGGRAPH_ENABLED=true이고 langgraph가 설치되어 있으면
        llm.py 내부 처리도 LangGraph workflow로 수행한다.
        비활성/미설치/실패 시 기존 선형 처리 함수로 fallback한다.
        """
        if self._should_use_langgraph():
            try:
                return self._answer_question_graph(question, chat_history, top_k)
            except Exception:
                # 그래프 구성/실행 실패가 서비스 장애로 이어지지 않도록 기존 처리로 복구한다.
                return self._answer_question_linear(question, chat_history, top_k)
        return self._answer_question_linear(question, chat_history, top_k)

    def _should_use_langgraph(self) -> bool:
        return bool(LLM_LANGGRAPH_ENABLED and StateGraph is not None)

    def _answer_question_graph(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 4,
    ) -> AgentResult:
        workflow = StateGraph(AgentWorkflowState)
        workflow.add_node("prepare", self._graph_prepare_node)
        workflow.add_node("rewrite", self._graph_rewrite_node)
        workflow.add_node("resolve", self._graph_resolve_node)
        workflow.add_node("guard", self._graph_guard_node)
        workflow.add_node("retrieve", self._graph_retrieve_node)
        workflow.add_node("respond", self._graph_respond_node)

        workflow.add_edge(START, "prepare")
        workflow.add_edge("prepare", "rewrite")
        workflow.add_edge("rewrite", "resolve")
        workflow.add_edge("resolve", "guard")
        workflow.add_conditional_edges(
            "guard",
            self._graph_after_guard,
            {
                "end": END,
                "retrieve": "retrieve",
            },
        )
        workflow.add_edge("retrieve", "respond")
        workflow.add_edge("respond", END)

        compiled = workflow.compile()
        initial_state: AgentWorkflowState = {
            "question": question,
            "chat_history": chat_history or [],
            "top_k": top_k,
            "debug_logs": [
                f"[LG 0] llm_langgraph_enabled = {LLM_LANGGRAPH_ENABLED}",
                f"[LC 1] feature_flags = {get_langchain_feature_flags()}",
            ],
        }
        final_state = compiled.invoke(initial_state)
        result = final_state.get("result")
        if not result:
            raise RuntimeError("LangGraph workflow did not return AgentResult.")
        return result

    def _graph_prepare_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        question = state.get("question", "")
        chat_history = state.get("chat_history", []) or []
        debug_logs = list(state.get("debug_logs", []))

        raw_question = normalize_whitespace(question)
        normalized_question = apply_dictionary_rewrite(raw_question)
        debug_logs.append("[LG 1] node = prepare")
        debug_logs.append(f"[PREP 1] whitespace_normalized = {raw_question}")
        debug_logs.append(f"[PREP 2] dictionary_standardized = {normalized_question}")

        intent_hint_before_rewrite = detect_intent(normalized_question)
        intent_hint_is_confident = is_confident_intent_hint(normalized_question, intent_hint_before_rewrite)
        intent_fallback_allowed = intent_hint_is_confident
        rewrite_intent_hint = "default"
        debug_logs.append(f"[PREP 3] raw_intent_hint_before_rewrite = {intent_hint_before_rewrite}")
        debug_logs.append(f"[PREP 3-0] intent_hint_is_confident = {intent_hint_is_confident}")
        debug_logs.append("[PREP 3-1] rewrite_intent_hint = default (intent is decided after rewrite)")

        direct_system_id = detect_system_id(normalized_question)
        if direct_system_id:
            system_id_hint = direct_system_id
            debug_logs.append("[PREP 3-2] system_id_hint_source = current_question")
        else:
            system_id_hint = None
            debug_logs.append("[PREP 3-2] system_id_hint_source = none_before_rewrite")

        rewrite_history = chat_history if is_followup_question(normalized_question) else []
        debug_logs.append(
            "[PREP 3-3] rewrite_history = enabled_for_followup"
            if rewrite_history
            else "[PREP 3-3] rewrite_history = disabled_for_current_question"
        )
        debug_logs.append(f"[PREP 4] resolved_system_id_hint_before_rewrite = {system_id_hint}")
        debug_logs.append(f"[PREP 5] resolved_intent_hint_before_rewrite = {rewrite_intent_hint}")

        return {
            **state,
            "raw_question": raw_question,
            "normalized_question": normalized_question,
            "intent_hint_before_rewrite": intent_hint_before_rewrite,
            "intent_hint_is_confident": intent_hint_is_confident,
            "intent_fallback_allowed": intent_fallback_allowed,
            "rewrite_intent_hint": rewrite_intent_hint,
            "system_id_hint": system_id_hint,
            "rewrite_history": rewrite_history,
            "debug_logs": debug_logs,
        }

    def _graph_rewrite_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        debug_logs = list(state.get("debug_logs", []))
        debug_logs.append("[LG 2] node = rewrite")
        rewritten_question, rewrite_logs = rewrite_question(
            question=state.get("normalized_question", ""),
            chat_history=state.get("rewrite_history", []) or [],
            config=self.chat_config,
            resolved_system_id=state.get("system_id_hint"),
            resolved_intent=state.get("rewrite_intent_hint", "default"),
        )
        debug_logs.extend(rewrite_logs)

        rewritten_standardized = apply_dictionary_rewrite(rewritten_question)
        if rewritten_standardized != rewritten_question:
            debug_logs.append(f"[REWRITE 8] post_dictionary_standardized = {rewritten_standardized}")
            rewritten_question = rewritten_standardized

        return {**state, "rewritten_question": rewritten_question, "debug_logs": debug_logs}

    def _graph_resolve_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        chat_history = state.get("chat_history", []) or []
        rewritten_question = state.get("rewritten_question", "")
        debug_logs = list(state.get("debug_logs", []))
        debug_logs.append("[LG 3] node = resolve")

        intent = detect_intent(rewritten_question)
        if intent == "default" and chat_history and is_followup_question(rewritten_question):
            previous_intent = detect_previous_user_intent(chat_history)
            if previous_intent != "default":
                intent = previous_intent
                debug_logs.append(f"[STEP 5-1] intent_fallback_to_history_followup = {previous_intent}")

        if (
            intent == "default"
            and state.get("intent_fallback_allowed")
            and state.get("intent_hint_before_rewrite") != "default"
        ):
            intent = str(state.get("intent_hint_before_rewrite"))
            debug_logs.append(f"[STEP 5-2] intent_fallback_to_confident_current_hint = {intent}")

        system_id = detect_system_id(rewritten_question) or state.get("system_id_hint")
        if not system_id and chat_history and is_followup_question(rewritten_question):
            system_id = detect_system_id_with_history(rewritten_question, chat_history)
            debug_logs.append("[STEP 5-3] system_id_fallback_to_history_followup = applied")
        elif not system_id:
            debug_logs.append("[STEP 5-3] system_id_fallback_to_history = skipped_not_followup")

        debug_logs.append(f"[STEP 5] detected_system_id = {system_id}")
        debug_logs.append(f"[STEP 6] detected_intent = {intent}")
        return {**state, "intent": intent, "system_id": system_id, "debug_logs": debug_logs}

    def _graph_guard_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        debug_logs = list(state.get("debug_logs", []))
        debug_logs.append("[LG 4] node = guard")
        intent = state.get("intent", "default")
        system_id = state.get("system_id")

        if intent == "default" and not system_id:
            debug_logs.append("[STEP 6-1] out_of_scope_detected = True")
            result = AgentResult(
                original_question=state.get("question", ""),
                normalized_question=state.get("normalized_question", ""),
                rewritten_question=state.get("rewritten_question", ""),
                system_id=None,
                intent="out_of_scope",
                answer=str(CONVERSATION_POLICY.get(
                    "out_of_scope_message",
                    "현재 에이전트의 지원 범위를 벗어난 질문입니다."
                )),
                render_type="text",
                debug_logs=debug_logs,
            )
            return {**state, "result": result, "debug_logs": debug_logs}

        system_required_intents = set(CONVERSATION_POLICY.get("system_required_intents", []))
        if intent in system_required_intents and not system_id:
            debug_logs.append("[STEP 6-2] missing_system_id = True")
            result = AgentResult(
                original_question=state.get("question", ""),
                normalized_question=state.get("normalized_question", ""),
                rewritten_question=state.get("rewritten_question", ""),
                system_id=None,
                intent=intent,
                answer=str(CONVERSATION_POLICY.get(
                    "unknown_system_message",
                    CONVERSATION_POLICY.get(
                        "missing_system_message",
                        "등록되지 않은 시스템입니다. 시스템명을 확인해주세요."
                    )
                )),
                render_type="text",
                debug_logs=debug_logs,
            )
            return {**state, "result": result, "debug_logs": debug_logs}

        render_type, graph_data, query_meta, structured_data = self._build_structured_payload(system_id, intent)
        where = self._build_where_filter(system_id, intent)
        debug_logs.append(f"[STEP 7] render_type = {render_type}")
        debug_logs.append(f"[STEP 8] where_filter = {where}")
        return {
            **state,
            "render_type": render_type,
            "graph_data": graph_data,
            "query_meta": query_meta,
            "structured_data": structured_data,
            "where": where,
            "debug_logs": debug_logs,
        }

    def _graph_after_guard(self, state: AgentWorkflowState) -> str:
        return "end" if state.get("result") else "retrieve"

    def _graph_retrieve_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        debug_logs = list(state.get("debug_logs", []))
        debug_logs.append("[LG 5] node = retrieve")
        search_query = state.get("rewritten_question", "")
        debug_logs.append(f"[STEP 9] search_query = {search_query}")
        if _use_langchain() and _env_flag("LANGCHAIN_RETRIEVER_ENABLED", "true"):
            debug_logs.append("[STEP 9-0] retriever_engine = langchain_chroma")
        else:
            debug_logs.append("[STEP 9-0] retriever_engine = chromadb_direct")

        search_result = retrieve_docs(
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
            query=search_query,
            top_k=int(state.get("top_k", 4)),
            where=state.get("where"),
        )
        search_result = compress_search_result(
            search_result=search_result,
            query=search_query,
            intent=state.get("intent", "default"),
            top_k=int(state.get("top_k", 4)),
        )
        debug_logs.append(
            "[STEP 9-1] retrieval_compression = enabled"
            if _env_flag("LANGCHAIN_RETRIEVAL_COMPRESSION_ENABLED", "false")
            else "[STEP 9-1] retrieval_compression = disabled"
        )

        documents = search_result.get("documents", [[]])[0]
        metadatas = search_result.get("metadatas", [[]])[0]
        debug_logs.append(f"[STEP 10] retrieved_doc_count = {len(documents)}")

        source_rows: List[Dict[str, Any]] = []
        for rank, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
            source_rows.append(
                {
                    "rank": rank,
                    "title": meta.get("title"),
                    "system_id": meta.get("system_id"),
                    "system_name": meta.get("system_name"),
                    "section": meta.get("section"),
                    "doc_level": meta.get("doc_level"),
                    "chunk_id": meta.get("chunk_id"),
                    "chunk_type": meta.get("chunk_type"),
                    "step": meta.get("step"),
                    "job_id": meta.get("job_id"),
                    "preview": (doc[:300] + "...") if len(doc) > 300 else doc,
                }
            )

        return {
            **state,
            "search_query": search_query,
            "search_result": search_result,
            "documents": documents,
            "metadatas": metadatas,
            "source_rows": source_rows,
            "debug_logs": debug_logs,
        }

    def _graph_respond_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        debug_logs = list(state.get("debug_logs", []))
        debug_logs.append("[LG 6] node = respond")
        question = state.get("question", "")
        normalized_question = state.get("normalized_question", "")
        rewritten_question = state.get("rewritten_question", "")
        system_id = state.get("system_id")
        intent = state.get("intent", "default")
        render_type = state.get("render_type", "text")
        graph_data = state.get("graph_data")
        query_meta = state.get("query_meta")
        structured_data = state.get("structured_data")
        source_rows = state.get("source_rows", []) or []
        documents = state.get("documents", []) or []
        search_result = state.get("search_result", {}) or {}
        chat_history = state.get("chat_history", []) or []

        if not documents and system_id and intent in {"overview", "batch_process", "batch_flow", "table_lineage"}:
            debug_logs.append("[STEP 10] filtered_retrieval_empty = fallback_to_structured_payload")
            if intent == "overview" and structured_data:
                result = AgentResult(
                    question, normalized_question, rewritten_question, system_id, intent,
                    build_overview_fallback(structured_data), render_type, graph_data, query_meta,
                    None, structured_data, [], debug_logs,
                )
                return {**state, "result": result, "debug_logs": debug_logs}
            if intent == "batch_process" and structured_data:
                result = AgentResult(
                    question, normalized_question, rewritten_question, system_id, intent,
                    build_batch_process_fallback(structured_data), render_type, graph_data, query_meta,
                    None, structured_data, [], debug_logs,
                )
                return {**state, "result": result, "debug_logs": debug_logs}

        route = resolve_response_route(
            intent=intent,
            render_type=render_type,
            has_graph=bool(graph_data),
            has_query_meta=bool(query_meta),
            query_meta=query_meta,
        )
        debug_logs.append(f"[STEP 11] response_route = {route.name}")

        if route.name == "graph" and graph_data:
            result = AgentResult(
                question, normalized_question, rewritten_question, system_id, intent,
                build_graph_answer(graph_data, intent), route.render_type, graph_data,
                None, None, None, source_rows, debug_logs,
            )
            return {**state, "result": result, "debug_logs": debug_logs}

        if route.name == "chart" and query_meta:
            result = AgentResult(
                question, normalized_question, rewritten_question, system_id, intent,
                build_chart_answer(query_meta), route.render_type, None, query_meta,
                route.realtime_mode, None, source_rows, debug_logs,
            )
            return {**state, "result": result, "debug_logs": debug_logs}

        if route.name == "table" and query_meta:
            result = AgentResult(
                question, normalized_question, rewritten_question, system_id, intent,
                build_table_answer(query_meta), route.render_type, None, query_meta,
                route.realtime_mode, None, source_rows, debug_logs,
            )
            return {**state, "result": result, "debug_logs": debug_logs}

        if intent == "overview" and structured_data:
            answer = build_overview_fallback(structured_data)
            debug_logs.append("[STEP 11] answer_generation = skipped (structured_overview_renderer_after_retrieval)")
            result = AgentResult(
                question, normalized_question, rewritten_question, system_id, intent,
                answer, render_type, graph_data, query_meta, None, structured_data,
                source_rows, debug_logs,
            )
            return {**state, "result": result, "debug_logs": debug_logs}

        if intent == "batch_process" and structured_data:
            answer = build_batch_process_fallback(structured_data)
            debug_logs.append("[STEP 11] answer_generation = skipped (structured_batch_renderer)")
            result = AgentResult(
                question, normalized_question, rewritten_question, system_id, intent,
                answer, render_type, graph_data, query_meta, None, structured_data,
                source_rows, debug_logs,
            )
            return {**state, "result": result, "debug_logs": debug_logs}

        system_prompt, prompt = build_answer_prompt(
            rewritten_question=rewritten_question,
            intent=intent,
            search_result=search_result,
            chat_history=chat_history,
            system_id=system_id,
        )

        try:
            debug_logs.append(f"[STEP 11] answer_generation_engine = {get_llm_engine_name()}")
            debug_logs.append("[STEP 12] answer_generation = started")
            answer = ollama_generate(
                prompt=prompt,
                system_prompt=system_prompt,
                config=self.chat_config,
            )
            if intent == "batch_process":
                answer = remove_repeated_step_sections(answer)
                debug_logs.append("[STEP 12-1] duplicate_step_cleanup = applied")
            debug_logs.append("[STEP 13] answer_generation = success")
        except Exception as e:
            debug_logs.append(f"[STEP 13] answer_generation = failed ({type(e).__name__}: {e})")
            if intent == "overview" and structured_data:
                answer = build_overview_fallback(structured_data)
                debug_logs.append("[STEP 14] fallback = overview_structured_payload")
            elif intent == "batch_process" and structured_data:
                answer = build_batch_process_fallback(structured_data)
                debug_logs.append("[STEP 14] fallback = batch_process_structured_payload")
            elif documents:
                answer = documents[0]
                debug_logs.append("[STEP 14] fallback = first_retrieved_document")
            else:
                answer = "관련 문서를 찾았지만 답변 생성에 실패했습니다."
                debug_logs.append("[STEP 14] fallback = generic_error_message")

        result = AgentResult(
            question, normalized_question, rewritten_question, system_id, intent,
            answer, render_type, graph_data, query_meta, None, structured_data,
            source_rows, debug_logs,
        )
        return {**state, "result": result, "debug_logs": debug_logs}

    def _answer_question_linear(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 4,
    ) -> AgentResult:
        chat_history = chat_history or []
        debug_logs: List[str] = []
        debug_logs.append(f"[LC 1] feature_flags = {get_langchain_feature_flags()}")

        # 실무 질의 처리 순서:
        # 1) 최소 정리: 공백만 정리
        # 2) dictionary: 업무 용어/시스템 alias 표준화
        # 3) LLM rewrite: 오타/구어체/의도 표현 보정
        # 4) rewrite 결과 기준 intent/system 재판별
        # 5) 검색
        raw_question = normalize_whitespace(question)
        normalized_question = apply_dictionary_rewrite(raw_question)
        debug_logs.append(f"[PREP 1] whitespace_normalized = {raw_question}")
        debug_logs.append(f"[PREP 2] dictionary_standardized = {normalized_question}")

        # rewrite 전 intent는 잠정 후보로만 기록한다.
        # 오타가 포함된 현재 질문에서 잡힌 약한 intent를 LLM rewrite에 고정하지 않는다.
        # 실제 라우팅 intent는 rewrite 이후 질문으로 다시 판별한다.
        intent_hint_before_rewrite = detect_intent(normalized_question)
        intent_hint_is_confident = is_confident_intent_hint(normalized_question, intent_hint_before_rewrite)
        intent_fallback_allowed = intent_hint_is_confident
        rewrite_intent_hint = "default"
        debug_logs.append(f"[PREP 3] raw_intent_hint_before_rewrite = {intent_hint_before_rewrite}")
        debug_logs.append(f"[PREP 3-0] intent_hint_is_confident = {intent_hint_is_confident}")
        debug_logs.append("[PREP 3-1] rewrite_intent_hint = default (intent is decided after rewrite)")

        direct_system_id = detect_system_id(normalized_question)
        if direct_system_id:
            system_id_hint = direct_system_id
            debug_logs.append("[PREP 3-2] system_id_hint_source = current_question")
        else:
            # rewrite 전에는 history system_id를 주입하지 않는다.
            # BB증권 같은 현재 질문의 시스템 오타가 이전 KKK 문맥으로 덮이는 것을 막기 위한 정책이다.
            # history system_id는 rewrite 이후에도 system_id가 없고, 질문이 후속질문일 때만 사용한다.
            system_id_hint = None
            debug_logs.append("[PREP 3-2] system_id_hint_source = none_before_rewrite")

        rewrite_history = chat_history if is_followup_question(normalized_question) else []
        if rewrite_history:
            debug_logs.append("[PREP 3-3] rewrite_history = enabled_for_followup")
        else:
            debug_logs.append("[PREP 3-3] rewrite_history = disabled_for_current_question")

        debug_logs.append(f"[PREP 4] resolved_system_id_hint_before_rewrite = {system_id_hint}")
        debug_logs.append(f"[PREP 5] resolved_intent_hint_before_rewrite = {rewrite_intent_hint}")

        rewritten_question, rewrite_logs = rewrite_question(
            question=normalized_question,
            chat_history=rewrite_history,
            config=self.chat_config,
            resolved_system_id=system_id_hint,
            resolved_intent=rewrite_intent_hint,
        )
        debug_logs.extend(rewrite_logs)

        rewritten_standardized = apply_dictionary_rewrite(rewritten_question)
        if rewritten_standardized != rewritten_question:
            debug_logs.append(f"[REWRITE 8] post_dictionary_standardized = {rewritten_standardized}")
            rewritten_question = rewritten_standardized

        intent = detect_intent(rewritten_question)

        if intent == "default" and chat_history and is_followup_question(rewritten_question):
            previous_intent = detect_previous_user_intent(chat_history)
            if previous_intent != "default":
                intent = previous_intent
                debug_logs.append(f"[STEP 5-1] intent_fallback_to_history_followup = {previous_intent}")

        if intent == "default" and intent_fallback_allowed and intent_hint_before_rewrite != "default":
            intent = intent_hint_before_rewrite
            debug_logs.append(f"[STEP 5-2] intent_fallback_to_confident_current_hint = {intent_hint_before_rewrite}")

        system_id = detect_system_id(rewritten_question) or system_id_hint
        if not system_id and chat_history and is_followup_question(rewritten_question):
            system_id = detect_system_id_with_history(rewritten_question, chat_history)
            debug_logs.append("[STEP 5-3] system_id_fallback_to_history_followup = applied")
        elif not system_id:
            debug_logs.append("[STEP 5-3] system_id_fallback_to_history = skipped_not_followup")

        debug_logs.append(f"[STEP 5] detected_system_id = {system_id}")
        debug_logs.append(f"[STEP 6] detected_intent = {intent}")

        if intent == "default" and not system_id:
            debug_logs.append("[STEP 6-1] out_of_scope_detected = True")
            return AgentResult(
                original_question=question,
                normalized_question=normalized_question,
                rewritten_question=rewritten_question,
                system_id=None,
                intent="out_of_scope",
                answer=str(CONVERSATION_POLICY.get(
                    "out_of_scope_message",
                    "현재 에이전트의 지원 범위를 벗어난 질문입니다."
                )),
                render_type="text",
                debug_logs=debug_logs,
            )

        system_required_intents = set(CONVERSATION_POLICY.get("system_required_intents", []))
        if intent in system_required_intents and not system_id:
            debug_logs.append("[STEP 6-2] missing_system_id = True")
            return AgentResult(
                original_question=question,
                normalized_question=normalized_question,
                rewritten_question=rewritten_question,
                system_id=None,
                intent=intent,
                answer=str(CONVERSATION_POLICY.get(
                    "unknown_system_message",
                    CONVERSATION_POLICY.get(
                        "missing_system_message",
                        "등록되지 않은 시스템입니다. 시스템명을 확인해주세요."
                    )
                )),
                render_type="text",
                debug_logs=debug_logs,
            )

        render_type, graph_data, query_meta, structured_data = self._build_structured_payload(system_id, intent)
        where = self._build_where_filter(system_id, intent)
        debug_logs.append(f"[STEP 7] render_type = {render_type}")
        debug_logs.append(f"[STEP 8] where_filter = {where}")

        search_query = rewritten_question
        debug_logs.append(f"[STEP 9] search_query = {search_query}")
        if _use_langchain() and _env_flag("LANGCHAIN_RETRIEVER_ENABLED", "true"):
            debug_logs.append("[STEP 9-0] retriever_engine = langchain_chroma")
        else:
            debug_logs.append("[STEP 9-0] retriever_engine = chromadb_direct")

        search_result = retrieve_docs(
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
            query=search_query,
            top_k=top_k,
            where=where,
        )

        search_result = compress_search_result(
            search_result=search_result,
            query=search_query,
            intent=intent,
            top_k=top_k,
        )
        if _env_flag("LANGCHAIN_RETRIEVAL_COMPRESSION_ENABLED", "false"):
            debug_logs.append("[STEP 9-1] retrieval_compression = enabled")
        else:
            debug_logs.append("[STEP 9-1] retrieval_compression = disabled")

        documents = search_result.get("documents", [[]])[0]
        metadatas = search_result.get("metadatas", [[]])[0]

        if not documents and system_id and intent in {"overview", "batch_process", "batch_flow", "table_lineage"}:
            debug_logs.append("[STEP 10] filtered_retrieval_empty = fallback_to_structured_payload")
            if intent == "overview" and structured_data:
                return AgentResult(
                    question,
                    normalized_question,
                    rewritten_question,
                    system_id,
                    intent,
                    build_overview_fallback(structured_data),
                    render_type,
                    graph_data,
                    query_meta,
                    None,
                    structured_data,
                    [],
                    debug_logs,
                )
            if intent == "batch_process" and structured_data:
                return AgentResult(
                    question,
                    normalized_question,
                    rewritten_question,
                    system_id,
                    intent,
                    build_batch_process_fallback(structured_data),
                    render_type,
                    graph_data,
                    query_meta,
                    None,
                    structured_data,
                    [],
                    debug_logs,
                )

        source_rows: List[Dict[str, Any]] = []
        debug_logs.append(f"[STEP 10] retrieved_doc_count = {len(documents)}")

        for rank, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
            source_rows.append(
                {
                    "rank": rank,
                    "title": meta.get("title"),
                    "system_id": meta.get("system_id"),
                    "system_name": meta.get("system_name"),
                    "section": meta.get("section"),
                    "doc_level": meta.get("doc_level"),
                    "chunk_id": meta.get("chunk_id"),
                    "chunk_type": meta.get("chunk_type"),
                    "step": meta.get("step"),
                    "job_id": meta.get("job_id"),
                    "preview": (doc[:300] + "...") if len(doc) > 300 else doc,
                }
            )

        route = resolve_response_route(
            intent=intent,
            render_type=render_type,
            has_graph=bool(graph_data),
            has_query_meta=bool(query_meta),
        )
        debug_logs.append(f"[STEP 11] response_route = {route.name}")

        if route.name == "graph" and graph_data:
            return AgentResult(
                question,
                normalized_question,
                rewritten_question,
                system_id,
                intent,
                build_graph_answer(graph_data, intent),
                route.render_type,
                graph_data,
                None,
                None,
                None,
                source_rows,
                debug_logs,
            )

        if route.name == "chart" and query_meta:
            return AgentResult(
                question,
                normalized_question,
                rewritten_question,
                system_id,
                intent,
                build_chart_answer(query_meta),
                route.render_type,
                None,
                query_meta,
                route.realtime_mode,
                None,
                source_rows,
                debug_logs,
            )

        if route.name == "table" and query_meta:
            return AgentResult(
                question,
                normalized_question,
                rewritten_question,
                system_id,
                intent,
                build_table_answer(query_meta),
                route.render_type,
                None,
                query_meta,
                route.realtime_mode,
                None,
                source_rows,
                debug_logs,
            )

        if intent == "overview" and structured_data:
            answer = build_overview_fallback(structured_data)
            debug_logs.append("[STEP 11] answer_generation = skipped (structured_overview_renderer_after_retrieval)")
            return AgentResult(
                question,
                normalized_question,
                rewritten_question,
                system_id,
                intent,
                answer,
                render_type,
                graph_data,
                query_meta,
                None,
                structured_data,
                source_rows,
                debug_logs,
            )

        if intent == "batch_process" and structured_data:
            answer = build_batch_process_fallback(structured_data)
            debug_logs.append("[STEP 11] answer_generation = skipped (structured_batch_renderer)")
            return AgentResult(
                question,
                normalized_question,
                rewritten_question,
                system_id,
                intent,
                answer,
                render_type,
                graph_data,
                query_meta,
                None,
                structured_data,
                source_rows,
                debug_logs,
            )

        system_prompt, prompt = build_answer_prompt(
            rewritten_question=rewritten_question,
            intent=intent,
            search_result=search_result,
            chat_history=chat_history,
            system_id=system_id,
        )

        try:
            debug_logs.append(f"[STEP 11] answer_generation_engine = {get_llm_engine_name()}")
            debug_logs.append("[STEP 12] answer_generation = started")
            answer = ollama_generate(
                prompt=prompt,
                system_prompt=system_prompt,
                config=self.chat_config,
            )
            if intent == "batch_process":
                answer = remove_repeated_step_sections(answer)
                debug_logs.append("[STEP 12-1] duplicate_step_cleanup = applied")
            debug_logs.append("[STEP 13] answer_generation = success")
        except Exception as e:
            debug_logs.append(f"[STEP 13] answer_generation = failed ({type(e).__name__}: {e})")
            if intent == "overview" and structured_data:
                answer = build_overview_fallback(structured_data)
                debug_logs.append("[STEP 14] fallback = overview_structured_payload")
            elif intent == "batch_process" and structured_data:
                answer = build_batch_process_fallback(structured_data)
                debug_logs.append("[STEP 14] fallback = batch_process_structured_payload")
            elif documents:
                answer = documents[0]
                debug_logs.append("[STEP 14] fallback = first_retrieved_document")
            else:
                answer = "관련 문서를 찾았지만 답변 생성에 실패했습니다."
                debug_logs.append("[STEP 14] fallback = generic_error_message")

        return AgentResult(
            question,
            normalized_question,
            rewritten_question,
            system_id,
            intent,
            answer,
            render_type,
            graph_data,
            query_meta,
            None,
            structured_data,
            source_rows,
            debug_logs,
        )
