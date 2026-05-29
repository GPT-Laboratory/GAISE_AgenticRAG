import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas import ChatRequest, ChatResponse, ToolCall
from app.chat import generate_chat_response, stream_chat_response

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    try:
        result = generate_chat_response(
            message=payload.message,
            history=[msg.model_dump() for msg in payload.history],
        )

        tool_calls = [ToolCall(**tool) for tool in result.get("tool_calls", [])]

        return ChatResponse(
            answer=result["answer"],
            tool_calls=tool_calls,
            chart_data=result.get("chart_data"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/chat/stream")
def chat_stream(payload: ChatRequest):
    """Server-Sent Events stream of the agent loop as it runs.

    Each message is `data: {json}\\n\\n`. Event types are documented in
    app.chat.stream_chat_response. The terminal `final` event carries the same
    payload as POST /chat (answer, tool_calls, chart_data).
    """
    history = [msg.model_dump() for msg in payload.history]

    def event_source():
        for event in stream_chat_response(payload.message, history):
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
