use axum::response::sse::Event;
use futures::StreamExt;
use reqwest::Client;
use tracing::debug;

/// Parse SSE events from an upstream llama-server streaming response.
///
/// Consumes the raw byte stream, buffers partial chunks, and yields
/// `axum::response::sse::Event` items. Terminates on `data: [DONE]`.
pub async fn stream_from_upstream(
    client: &Client,
    url: &str,
    body: &serde_json::Value,
) -> anyhow::Result<impl futures::Stream<Item = Result<Event, anyhow::Error>>> {
    let response = client
        .post(url)
        .json(body)
        .send()
        .await?
        .error_for_status()?;

    debug!("SSE stream connected to upstream");

    let stream = async_stream::stream! {
        let mut bytes_stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = bytes_stream.next().await {
            let chunk = chunk?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Parse complete SSE events from the buffer.
            // SSE events are delimited by double newlines.
            while let Some(pos) = buffer.find("\n\n") {
                let event_text = buffer[..pos].to_string();
                buffer = buffer[pos + 2..].to_string();

                for line in event_text.lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            debug!("SSE stream received [DONE]");
                            return;
                        }
                        yield Ok(Event::default().data(data));
                    }
                }
            }
        }

        debug!("SSE upstream stream ended");
    };

    Ok(stream)
}

/// Parse SSE data lines from a raw text buffer.
///
/// Returns parsed events and the remaining unparsed buffer.
/// Used for testing the parsing logic in isolation.
pub fn parse_sse_buffer(buffer: &str) -> (Vec<SseEvent>, String) {
    let mut events = Vec::new();
    let mut remaining = buffer.to_string();

    while let Some(pos) = remaining.find("\n\n") {
        let event_text = remaining[..pos].to_string();
        remaining = remaining[pos + 2..].to_string();

        for line in event_text.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    events.push(SseEvent::Done);
                } else {
                    events.push(SseEvent::Data(data.to_string()));
                }
            }
        }
    }

    (events, remaining)
}

/// Parsed SSE event for testing.
#[derive(Debug, PartialEq, Eq)]
pub enum SseEvent {
    Data(String),
    Done,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sse_single_event() {
        let input = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n";
        let (events, remaining) = parse_sse_buffer(input);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], SseEvent::Data(d) if d.contains("Hello")));
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_sse_multiple_events() {
        let input = "data: {\"content\":\"A\"}\n\ndata: {\"content\":\"B\"}\n\n";
        let (events, remaining) = parse_sse_buffer(input);
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], SseEvent::Data(d) if d.contains("A")));
        assert!(matches!(&events[1], SseEvent::Data(d) if d.contains("B")));
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_sse_done_sentinel() {
        let input = "data: {\"content\":\"Hi\"}\n\ndata: [DONE]\n\n";
        let (events, remaining) = parse_sse_buffer(input);
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], SseEvent::Data(_)));
        assert_eq!(events[1], SseEvent::Done);
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_sse_partial_buffer() {
        // Incomplete event — no double newline yet
        let input = "data: {\"content\":\"partial\"}\n";
        let (events, remaining) = parse_sse_buffer(input);
        assert!(events.is_empty());
        assert_eq!(remaining, input);
    }

    #[test]
    fn test_parse_sse_partial_then_complete() {
        // First call: partial
        let (events1, remaining1) = parse_sse_buffer("data: {\"a\":1}\n");
        assert!(events1.is_empty());

        // Second call: complete after appending more data
        let combined = format!("{remaining1}\ndata: {{\"b\":2}}\n\n");
        let (events2, remaining2) = parse_sse_buffer(&combined);
        assert_eq!(events2.len(), 2);
        assert!(remaining2.is_empty());
    }

    #[test]
    fn test_parse_sse_ignores_non_data_lines() {
        let input = "event: message\ndata: {\"content\":\"Hi\"}\nid: 1\n\n";
        let (events, _) = parse_sse_buffer(input);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], SseEvent::Data(d) if d.contains("Hi")));
    }
}
