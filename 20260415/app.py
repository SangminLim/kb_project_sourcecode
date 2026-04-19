import uuid
import time
import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_response

load_dotenv()

st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")

st.title("🤖 소득세 챗봇")
st.caption("소득세에 관련된 모든것을 답해드립니다!")

# 세션 초기화
if "message_list" not in st.session_state:
    st.session_state.message_list = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# 기존 메시지 출력
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 입력
user_question = st.chat_input(
    placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"
)

if user_question:
    print("\n====================", flush=True)
    print("[DEBUG] 사용자 질문 입력됨", flush=True)
    print(f"[DEBUG] 질문: {user_question}", flush=True)
    print(f"[DEBUG] session_id: {st.session_state.session_id}", flush=True)

    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.write(user_question)

    st.session_state.message_list.append({
        "role": "user",
        "content": user_question
    })

    # 응답 생성 시작
    start_time = time.time()
    print(f"[TIME] 요청 시작: {start_time:.4f}", flush=True)

    with st.spinner("답변을 생성하는 중입니다..."):
        try:
            ai_response = get_ai_response(
                question=user_question,
                session_id=st.session_state.session_id,
                message_list=st.session_state.message_list,
            )

            end_time = time.time()
            print(f"[TIME] 전체 응답 시간: {end_time - start_time:.2f}초", flush=True)

            with st.chat_message("assistant"):
                st.write(ai_response)

            st.session_state.message_list.append({
                "role": "assistant",
                "content": ai_response
            })

        except Exception as e:
            print(f"[ERROR] {e}", flush=True)
            st.error(f"에러 발생: {e}")