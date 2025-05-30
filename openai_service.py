import os
import json
import logging
from typing import List, Dict
from openai import OpenAI

logger = logging.getLogger(__name__)

class OpenAIServiceError(Exception):
    """Custom exception for OpenAI service errors"""
    pass

# Initialize OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def generate_topic_timestamps(srt_entries: List[Dict[str, str]], context: str = None) -> List[str]:
    """
    Generate topic-based timestamps using OpenAI API.
    
    Args:
        srt_entries (List[Dict[str, str]]): List of SRT entries with start, end, text
        context (str, optional): Additional context for better topic detection
        
    Returns:
        List[str]: List of formatted timestamps with topic descriptions
        
    Raises:
        OpenAIServiceError: If OpenAI API call fails or returns invalid response
    """
    if not openai_client:
        raise OpenAIServiceError("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
    
    try:
        # Format SRT content for analysis
        formatted_content = format_srt_for_openai(srt_entries)
        
        # Log content statistics
        content_length = len(formatted_content)
        content_words = len(formatted_content.split())
        logger.info(f"Formatted content stats: {content_length} characters, ~{content_words} words")
        
        # Check if content might be too large (rough estimate: 1 token ≈ 4 characters)
        estimated_tokens = content_length // 4
        logger.info(f"Estimated input tokens: {estimated_tokens}")
        
        # Get the last timestamp to check video duration
        if srt_entries:
            last_entry = srt_entries[-1]
            logger.info(f"Video duration: First entry at {srt_entries[0]['start']}, Last entry at {last_entry['end']}")
        
        # Use chunking strategy for large videos (lowered threshold significantly)
        if estimated_tokens > 50000 or len(srt_entries) > 1000:  # Much more conservative limit
            logger.info(f"Large video detected ({estimated_tokens} tokens, {len(srt_entries)} entries). Using chunking strategy.")
            return _generate_timestamps_chunked(srt_entries, context)
        
        # For smaller videos, use single request
        return _generate_timestamps_single(srt_entries, context, formatted_content)
    
    except OpenAIServiceError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in OpenAI service: {str(e)}")
        raise OpenAIServiceError(f"Failed to generate timestamps: {str(e)}")

def _generate_timestamps_single(srt_entries: List[Dict[str, str]], context: str = None, formatted_content: str = "") -> List[str]:
    """Generate timestamps for smaller videos in a single API call."""
    
    if not openai_client:
        raise OpenAIServiceError("OpenAI client not initialized")
    
    # Build the system prompt
    system_prompt = """You are an expert at analyzing video content and identifying topic segments from subtitles. 
Your task is to analyze subtitle text with timestamps and identify distinct topic segments or content changes.

Return a JSON object with a "timestamps" array containing objects with "time" and "description" fields.
Each timestamp should mark the beginning of a new topic or significant content shift.

Guidelines:
- Use the format "H:MM:SS" for timestamps (e.g., "0:00:15", "0:18:11", "1:23:45")
- Descriptions should be brief but descriptive and detailed as possible (under 115 characters)
- Focus on meaningful content transitions, not minor topic shifts
- Analyze the ENTIRE content provided - do not stop early
- Generate timestamps throughout the full duration of the content

Example output format:
{
  "timestamps": [
    {"time": "0:00:00", "description": "Introduction and sponsor mentions"},
    {"time": "0:02:15", "description": "Main topic discussion begins"},
    {"time": "0:15:30", "description": "Technical details and examples"},
    {"time": "0:28:45", "description": "Q&A and closing remarks"}
  ]
}"""

    # Build the user prompt
    user_prompt = f"Analyze the following subtitle content and identify topic segments:\n\n{formatted_content}"
    
    if context:
        user_prompt += f"\n\nAdditional context: {context}"
    
    user_prompt += "\n\nIMPORTANT: Analyze the COMPLETE content provided. Generate timestamps throughout the entire duration, not just the beginning."
    
    logger.debug(f"Sending single request to OpenAI with {len(srt_entries)} subtitle entries")
    
    # Make API call to OpenAI
    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_completion_tokens=15000,
        response_format={"type": "json_object"}
    )
    
    response_content = response.choices[0].message.content
    if not response_content:
        raise OpenAIServiceError("Empty response from OpenAI")
    
    return _parse_openai_response(response_content)

def _generate_timestamps_chunked(srt_entries: List[Dict[str, str]], context: str = None) -> List[str]:
    """Generate timestamps for large videos using chunking strategy."""
    
    if not openai_client:
        raise OpenAIServiceError("OpenAI client not initialized")
    
    # Calculate chunk size (aim for ~80k tokens per chunk with overlap)
    target_chars_per_chunk = 300000  # ~75k tokens
    overlap_chars = 40000  # ~10k tokens overlap
    
    all_timestamps = []
    chunks = _create_content_chunks(srt_entries, target_chars_per_chunk, overlap_chars)
    
    logger.info(f"Processing {len(chunks)} chunks for large video")
    
    for i, (chunk_entries, chunk_start_time, chunk_end_time) in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)} ({chunk_start_time} - {chunk_end_time})")
        
        formatted_chunk = format_srt_for_openai(chunk_entries)
        
        # Build chunk-specific system prompt
        system_prompt = f"""You are an expert at analyzing video content and identifying topic segments from subtitles. 
This is chunk {i+1} of {len(chunks)} from a longer video (time range: {chunk_start_time} - {chunk_end_time}).

Return a JSON object with a "timestamps" array containing objects with "time" and "description" fields.
Each timestamp should mark the beginning of a new topic or significant content shift.

Guidelines:
- Use the format "H:MM:SS" for timestamps (e.g., "0:00:15", "0:18:11", "1:23:45")
- Descriptions should be brief but descriptive (under 115 characters)
- Focus on meaningful content transitions within this time segment
- Only generate timestamps that fall within the time range {chunk_start_time} - {chunk_end_time}
- Analyze the ENTIRE chunk content provided

Example output format:
{{
  "timestamps": [
    {{"time": "0:00:00", "description": "Topic description"}},
    {{"time": "0:15:30", "description": "Another topic description"}}
  ]
}}"""

        user_prompt = f"Analyze this portion of subtitle content (time range {chunk_start_time} - {chunk_end_time}):\n\n{formatted_chunk}"
        
        if context:
            user_prompt += f"\n\nVideo context: {context}"
        
        user_prompt += f"\n\nGenerate timestamps only for content within {chunk_start_time} - {chunk_end_time}."
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=8000,
                response_format={"type": "json_object"}
            )
            
            response_content = response.choices[0].message.content
            if not response_content:
                logger.warning(f"Empty response for chunk {i+1}")
                continue
                
            chunk_timestamps = _parse_openai_response(response_content)
            all_timestamps.extend(chunk_timestamps)
            
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {str(e)}")
            # Continue with other chunks rather than failing completely
            continue
    
    # Remove duplicates and sort by time
    unique_timestamps = _deduplicate_timestamps(all_timestamps)
    
    logger.info(f"Generated {len(unique_timestamps)} total timestamps from {len(chunks)} chunks")
    return unique_timestamps

def _create_content_chunks(srt_entries: List[Dict[str, str]], target_chars: int, overlap_chars: int):
    """Create overlapping chunks of SRT entries."""
    chunks = []
    current_chunk = []
    current_chars = 0
    
    i = 0
    while i < len(srt_entries):
        entry = srt_entries[i]
        entry_text = f"[{entry['start']}] {entry['text']}\n"
        entry_chars = len(entry_text)
        
        # If adding this entry would exceed target, create a chunk
        if current_chars + entry_chars > target_chars and current_chunk:
            chunk_start = current_chunk[0]['start']
            chunk_end = current_chunk[-1]['end']
            chunks.append((current_chunk.copy(), chunk_start, chunk_end))
            
            # Start new chunk with overlap
            overlap_start = max(0, len(current_chunk) - int(len(current_chunk) * 0.1))  # 10% overlap
            current_chunk = current_chunk[overlap_start:]
            current_chars = sum(len(f"[{e['start']}] {e['text']}\n") for e in current_chunk)
        
        current_chunk.append(entry)
        current_chars += entry_chars
        i += 1
    
    # Add final chunk if it has content
    if current_chunk:
        chunk_start = current_chunk[0]['start']
        chunk_end = current_chunk[-1]['end']
        chunks.append((current_chunk, chunk_start, chunk_end))
    
    return chunks

def _parse_openai_response(response_content: str) -> List[str]:
    """Parse OpenAI response and return formatted timestamps."""
    logger.debug(f"Received response from OpenAI: {response_content[:200]}...")
    
    try:
        result = json.loads(response_content)
        timestamps_data = result.get("timestamps", [])
        
        if not timestamps_data:
            raise OpenAIServiceError("No timestamps found in OpenAI response")
        
        # Format timestamps for display
        formatted_timestamps = []
        for item in timestamps_data:
            time = item.get("time", "")
            description = item.get("description", "")
            
            if time and description:
                formatted_timestamps.append(f"{time} - {description}")
            else:
                logger.warning(f"Skipping malformed timestamp entry: {item}")
        
        return formatted_timestamps
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse OpenAI JSON response: {str(e)}")
        raise OpenAIServiceError(f"Invalid JSON response from OpenAI: {str(e)}")

def _deduplicate_timestamps(timestamps: List[str]) -> List[str]:
    """Remove duplicate timestamps and sort by time."""
    seen = set()
    unique = []
    
    # Parse and sort timestamps
    parsed_timestamps = []
    for ts in timestamps:
        if " - " in ts:
            time_part, desc_part = ts.split(" - ", 1)
            parsed_timestamps.append((time_part.strip(), desc_part.strip(), ts))
    
    # Sort by time
    def time_to_seconds(time_str):
        try:
            parts = time_str.split(":")
            if len(parts) == 3:
                h, m, s = map(int, parts)
                return h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m, s = map(int, parts)
                return m * 60 + s
            else:
                return 0
        except:
            return 0
    
    parsed_timestamps.sort(key=lambda x: time_to_seconds(x[0]))
    
    # Remove duplicates (same time within 30 seconds)
    for time_part, desc_part, full_ts in parsed_timestamps:
        time_key = time_to_seconds(time_part) // 30  # Group by 30-second intervals
        if time_key not in seen:
            seen.add(time_key)
            unique.append(full_ts)
    
    return unique

def format_srt_for_openai(srt_entries: List[Dict[str, str]]) -> str:
    """
    Format SRT entries for OpenAI analysis.
    
    Args:
        srt_entries (List[Dict[str, str]]): List of SRT entries
        
    Returns:
        str: Formatted content for OpenAI analysis
    """
    formatted_lines = []
    
    for entry in srt_entries:
        # Include timestamp and text for analysis
        formatted_lines.append(f"[{entry['start']}] {entry['text']}")
    
    return '\n'.join(formatted_lines)
