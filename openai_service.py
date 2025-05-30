import os
import json
import logging
from typing import List, Dict, Tuple
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


def calculate_time_based_chunks(srt_entries: List[Dict[str, str]]) -> List[int]:
    """
    Calculate chunk boundaries based on hour intervals, adjusted for topic boundaries.
    
    Args:
        srt_entries (List[Dict[str, str]]): List of SRT entries
        
    Returns:
        List[int]: List of indices where chunks should start
    """
    if not srt_entries:
        return [0]
    
    chunk_boundaries = [0]  # Always start with the first entry
    target_interval = 1800  # 30 minutes in seconds (smaller chunks)
    
    start_time = time_to_seconds(srt_entries[0]['start'])
    
    for i, entry in enumerate(srt_entries):
        current_time = time_to_seconds(entry['start'])
        time_since_last_chunk = current_time - start_time
        
        # Check if we've passed an hour mark
        if time_since_last_chunk >= target_interval:
            # Look for a good break point within the next 10 minutes
            search_end = min(i + 50, len(srt_entries))  # Roughly 10 minutes of entries
            best_break = i
            
            for j in range(i, search_end):
                if j >= len(srt_entries):
                    break
                    
                text = srt_entries[j]['text'].strip()
                # Look for sentence endings or natural breaks
                if text.endswith(('.', '!', '?', '。', '！', '？')):
                    best_break = j + 1
                    break
                
                # Look for time gaps (potential topic breaks)
                if j > 0:
                    try:
                        gap_time = time_to_seconds(srt_entries[j]['start'])
                        prev_time = time_to_seconds(srt_entries[j-1]['end'])
                        if gap_time - prev_time > 2.0:  # 2+ second gap
                            best_break = j
                            break
                    except:
                        pass
            
            chunk_boundaries.append(best_break)
            start_time = time_to_seconds(srt_entries[best_break]['start']) if best_break < len(srt_entries) else current_time
    
    logger.debug(f"Created {len(chunk_boundaries)} time-based chunk boundaries at hours: {[time_to_seconds(srt_entries[idx]['start'])/3600 for idx in chunk_boundaries if idx < len(srt_entries)]}")
    return chunk_boundaries


def calculate_chunk_size(srt_entries: List[Dict[str, str]]) -> int:
    """
    Calculate optimal chunk size based on content length to stay within token limits.
    This is used as a fallback when time-based chunking isn't suitable.
    
    Args:
        srt_entries (List[Dict[str, str]]): List of SRT entries
        
    Returns:
        int: Optimal chunk size (number of entries per chunk)
    """
    total_entries = len(srt_entries)
    
    # Estimate average characters per entry (including formatting)
    sample_size = min(10, total_entries)
    total_chars = sum(len(entry['text']) + len(entry['start']) + 10 for entry in srt_entries[:sample_size])
    avg_chars_per_entry = total_chars / sample_size if sample_size > 0 else 100
    
    # Conservative token estimate: ~4 chars per token, target ~3000 tokens per chunk
    # This leaves room for system prompt and response
    target_chars_per_chunk = 12000
    chunk_size = max(10, int(target_chars_per_chunk / avg_chars_per_entry))
    
    # Cap chunk size to reasonable limits
    chunk_size = min(chunk_size, max(50, total_entries // 4))
    
    logger.debug(f"Calculated chunk size: {chunk_size} entries (avg {avg_chars_per_entry:.1f} chars/entry)")
    return chunk_size


def time_to_seconds(time_str: str) -> float:
    """Convert HH:MM:SS.mmm format to seconds."""
    try:
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_parts = parts[2].split('.')
        seconds = int(seconds_parts[0])
        milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    except:
        return 0.0


def create_smart_chunks(srt_entries: List[Dict[str, str]], chunk_size: int = None) -> List[Tuple[List[Dict[str, str]], int, int]]:
    """
    Create smart chunks based on time intervals (around every hour) with topic-aware boundaries.
    
    Args:
        srt_entries (List[Dict[str, str]]): List of SRT entries
        chunk_size (int, optional): Fallback chunk size if time-based chunking isn't suitable
        
    Returns:
        List[Tuple[List[Dict[str, str]], int, int]]: List of (chunk_entries, start_idx, end_idx)
    """
    if not srt_entries:
        return []
    
    # Get video duration to decide chunking strategy
    first_time = time_to_seconds(srt_entries[0]['start'])
    last_time = time_to_seconds(srt_entries[-1]['end'])
    video_duration = last_time - first_time
    
    # Use time-based chunking for videos longer than 30 minutes (more aggressive)
    if video_duration > 1800:  # 30 minutes
        chunk_boundaries = calculate_time_based_chunks(srt_entries)
        chunks = []
        overlap_size = 3  # Smaller overlap for faster processing
        
        for i in range(len(chunk_boundaries)):
            start_idx = chunk_boundaries[i]
            
            # Determine end index
            if i + 1 < len(chunk_boundaries):
                end_idx = chunk_boundaries[i + 1]
            else:
                end_idx = len(srt_entries)
            
            # Add overlap from previous chunk (except for first chunk)
            if i > 0:
                start_idx = max(0, start_idx - overlap_size)
            
            chunk_entries = srt_entries[start_idx:end_idx]
            if chunk_entries:
                # Further split large chunks if they're still too big
                if len(chunk_entries) > 200:  # Split very large chunks
                    mid_point = len(chunk_entries) // 2
                    chunks.append((chunk_entries[:mid_point], start_idx, start_idx + mid_point - 1))
                    chunks.append((chunk_entries[mid_point:], start_idx + mid_point, end_idx - 1))
                else:
                    chunks.append((chunk_entries, start_idx, end_idx - 1))
        
        logger.debug(f"Created {len(chunks)} time-based chunks for {video_duration/3600:.1f} hour video")
        return chunks
    
    # Fall back to size-based chunking for shorter videos
    if chunk_size is None:
        chunk_size = calculate_chunk_size(srt_entries)
    
    if len(srt_entries) <= chunk_size:
        return [(srt_entries, 0, len(srt_entries) - 1)]
    
    chunks = []
    overlap_size = max(2, chunk_size // 10)  # 10% overlap, minimum 2 entries
    i = 0
    
    while i < len(srt_entries):
        # Determine chunk end
        chunk_end = min(i + chunk_size, len(srt_entries))
        
        # Try to find a better break point near the end of the chunk
        if chunk_end < len(srt_entries):
            # Look for sentence endings within the last 20% of the chunk
            search_start = max(i + int(chunk_size * 0.8), i + 1)
            search_end = min(chunk_end + 5, len(srt_entries))
            
            best_break = chunk_end
            for j in range(search_start, search_end):
                if j >= len(srt_entries):
                    break
                    
                text = srt_entries[j]['text'].strip()
                # Look for sentence endings
                if text.endswith(('.', '!', '?', '。', '！', '？')):
                    best_break = j + 1
                    break
                
                # Look for significant time gaps (potential topic breaks)
                if j > 0:
                    try:
                        current_time = time_to_seconds(srt_entries[j]['start'])
                        prev_time = time_to_seconds(srt_entries[j-1]['end'])
                        if current_time - prev_time > 3.0:  # 3+ second gap
                            best_break = j
                            break
                    except:
                        pass
            
            chunk_end = best_break
        
        # Create chunk
        chunk_entries = srt_entries[i:chunk_end]
        chunks.append((chunk_entries, i, chunk_end - 1))
        
        # Move to next chunk with overlap
        if chunk_end >= len(srt_entries):
            break
        i = max(chunk_end - overlap_size, i + 1)
    
    logger.debug(f"Created {len(chunks)} size-based chunks with overlap")
    return chunks


def merge_chunk_results(chunk_results: List[List[str]], chunk_info: List[Tuple[List[Dict[str, str]], int, int]]) -> List[str]:
    """
    Merge timestamp results from multiple chunks, removing duplicates and sorting.
    
    Args:
        chunk_results (List[List[str]]): Results from each chunk
        chunk_info (List[Tuple]): Information about each chunk
        
    Returns:
        List[str]: Merged and deduplicated timestamps
    """
    all_timestamps = []
    seen_times = set()
    
    for chunk_idx, timestamps in enumerate(chunk_results):
        for timestamp in timestamps:
            # Extract time from timestamp (format: "H:MM:SS - Description")
            if ' - ' in timestamp:
                time_part = timestamp.split(' - ')[0].strip()
                
                # Avoid near-duplicate timestamps (within 10 seconds)
                time_seconds = time_to_seconds(time_part + '.000')
                is_duplicate = any(abs(time_seconds - seen_time) < 10 for seen_time in seen_times)
                
                if not is_duplicate:
                    all_timestamps.append(timestamp)
                    seen_times.add(time_seconds)
    
    # Sort by timestamp
    def sort_key(ts):
        try:
            time_part = ts.split(' - ')[0].strip()
            return time_to_seconds(time_part + '.000')
        except:
            return 0
    
    all_timestamps.sort(key=sort_key)
    logger.debug(f"Merged {len(all_timestamps)} unique timestamps from {len(chunk_results)} chunks")
    return all_timestamps


def _make_openai_request(system_prompt: str, user_prompt: str) -> List[str]:
    """Make a request to OpenAI and parse the response with retry logic."""
    import time
    
    if not openai_client:
        raise OpenAIServiceError("OpenAI client not initialized. Please check your OPENAI_API_KEY.")
    
    max_retries = 3
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=2000,
                response_format={"type": "json_object"},
                timeout=30  # Further reduced timeout
            )
            
            response_content = response.choices[0].message.content
            if not response_content:
                return []
            
            try:
                result = json.loads(response_content)
                timestamps_data = result.get("timestamps", [])
                
                # Format timestamps for display
                formatted_timestamps = []
                for item in timestamps_data:
                    time_str = item.get("time", "")
                    description = item.get("description", "")
                    
                    if time_str and description:
                        formatted_timestamps.append(f"{time_str} - {description}")
                    else:
                        logger.warning(f"Skipping malformed timestamp entry: {item}")
                
                return formatted_timestamps
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI JSON response: {str(e)}")
                raise OpenAIServiceError(f"Invalid JSON response from OpenAI: {str(e)}")
                
        except Exception as e:
            error_msg = str(e).lower()
            
            if attempt < max_retries - 1:
                if "rate limit" in error_msg or "timeout" in error_msg or "connection" in error_msg:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"OpenAI request failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {str(e)}")
                    time.sleep(delay)
                    continue
            
            # Handle specific error types
            if "rate limit" in error_msg:
                raise OpenAIServiceError("Rate limit exceeded. Please try again in a few minutes.")
            elif "insufficient_quota" in error_msg or "quota" in error_msg:
                raise OpenAIServiceError("OpenAI API quota exceeded. Please check your account.")
            elif "authentication" in error_msg or "invalid" in error_msg:
                raise OpenAIServiceError("Invalid OpenAI API key. Please check your OPENAI_API_KEY.")
            elif "timeout" in error_msg or "connection" in error_msg:
                raise OpenAIServiceError("Connection timeout. Please check your internet connection and try again.")
            else:
                raise OpenAIServiceError(f"OpenAI API error: {str(e)}")
    
    raise OpenAIServiceError("Failed to connect to OpenAI after multiple attempts.")


def _process_single_chunk(srt_entries: List[Dict[str, str]], context: str = None) -> List[str]:
    """Process a single chunk (original behavior for small files)."""
    formatted_content = format_srt_for_openai(srt_entries)
    
    system_prompt = """You are an expert at analyzing video content and identifying topic segments from subtitles. 
Your task is to analyze subtitle text with timestamps and identify distinct topic segments or content changes.

Return a JSON object with a "timestamps" array containing objects with "time" and "description" fields.
Each timestamp should mark the beginning of a new topic or significant content shift.

Guidelines:
- Use the format "H:MM:SS" for timestamps (e.g., "0:00:15", "0:18:11", "1:23:45")
- Descriptions should be brief but descriptive and detailed as possible. Use overall show context to get idea of whats being talked about. (under 115 characters)
- Focus on meaningful content transitions, not minor topic shifts
- Number of segments should be reasonable (around 5-15 per hour of content)
- Start with "0:00:00" if the content begins immediately

Example output format:
{
  "timestamps": [
    {"time": "0:00:00", "description": "Woody introduces the show and guest, Taylor the sponsors - New mystery sponsor!"},
    {"time": "0:02:15", "description": "Why Ed would never do a 'Top Gear' style show & how brand deals ruined content"},
    {"time": "0:15:30", "description": "Movie/TV talk: Tom Cruise, Yellowstone and why every actor is CGI deaged in 2025"}
  ]
}"""

    user_prompt = f"Analyze the following subtitle content and identify topic segments:\n\n{formatted_content}"
    
    if context:
        user_prompt += f"\n\nAdditional context: {context}"
    
    user_prompt += "\n\nGenerate topic-based timestamps in JSON format as specified."
    
    return _make_openai_request(system_prompt, user_prompt)


def _process_chunk_with_context(chunk_entries: List[Dict[str, str]], context: str, chunk_context: str, chunk_num: int, total_chunks: int) -> List[str]:
    """Process a chunk with context from previous chunks."""
    formatted_content = format_srt_for_openai(chunk_entries)
    
    system_prompt = """You are an expert at analyzing video content and identifying topic segments from subtitles. 
Your task is to analyze subtitle text with timestamps and identify distinct topic segments or content changes.

Return a JSON object with a "timestamps" array containing objects with "time" and "description" fields.
Each timestamp should mark the beginning of a new topic or significant content shift.

Guidelines:
- Use the format "H:MM:SS" for timestamps (e.g., "0:00:15", "0:18:11", "1:23:45")
- Descriptions should be brief but descriptive and detailed as possible. Use overall show context to get idea of whats being talked about. (under 115 characters)
- Focus on major topic changes rather than minor details
- This is part of a longer video, so focus on NEW topics that begin in this section"""

    user_prompt = f"Analyze this section of subtitle content (part {chunk_num} of {total_chunks}) and identify topic segments:\n\n{formatted_content}"
    
    if chunk_context:
        user_prompt = f"Previous context: {chunk_context}\n\n{user_prompt}"
    
    if context:
        user_prompt += f"\n\nAdditional context: {context}"
    
    user_prompt += f"\n\nThis is part {chunk_num} of {total_chunks} from a longer video. Focus on new topics that begin in this section.\nGenerate topic-based timestamps in JSON format as specified."
    
    return _make_openai_request(system_prompt, user_prompt)


def generate_topic_timestamps(srt_entries: List[Dict[str, str]], context: str = None) -> List[str]:
    """
    Generate topic-based timestamps using OpenAI API with smart chunking for large files.
    
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
        # Create smart chunks (time-based for long videos, size-based for short ones)
        chunks = create_smart_chunks(srt_entries)
        
        logger.debug(f"Processing {len(srt_entries)} entries in {len(chunks)} chunks")
        
        # For small files, use the original single-request approach
        if len(chunks) == 1:
            return _process_single_chunk(srt_entries, context)
        
        # Process multiple chunks with context
        chunk_results = []
        chunk_context = ""
        
        for chunk_idx, (chunk_entries, start_idx, end_idx) in enumerate(chunks):
            try:
                # Calculate chunk duration for logging
                chunk_start_time = time_to_seconds(chunk_entries[0]['start']) / 3600 if chunk_entries else 0
                chunk_end_time = time_to_seconds(chunk_entries[-1]['end']) / 3600 if chunk_entries else 0
                
                logger.debug(f"Processing chunk {chunk_idx + 1}/{len(chunks)} (entries {start_idx+1}-{end_idx+1}, time {chunk_start_time:.1f}h-{chunk_end_time:.1f}h)")
                
                # Process chunk with context from previous chunks
                chunk_timestamps = _process_chunk_with_context(
                    chunk_entries, context, chunk_context, chunk_idx + 1, len(chunks)
                )
                
                chunk_results.append(chunk_timestamps)
                
                # Update context for next chunk (last few topics)
                if chunk_timestamps:
                    recent_topics = [ts.split(' - ')[1] for ts in chunk_timestamps[-2:]]
                    chunk_context = f"Recent topics: {', '.join(recent_topics)}"
                
                logger.debug(f"Chunk {chunk_idx + 1} generated {len(chunk_timestamps)} timestamps")
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx + 1}: {str(e)}")
                # Continue with other chunks even if one fails
                chunk_results.append([])
        
        # Merge results from all chunks
        if not any(chunk_results):
            raise OpenAIServiceError("No timestamps could be generated from any chunk")
        
        final_timestamps = merge_chunk_results(chunk_results, chunks)
        
        if not final_timestamps:
            raise OpenAIServiceError("No valid timestamps could be extracted")
        
        logger.info(f"Successfully generated {len(final_timestamps)} total timestamps from {len(chunks)} chunks")
        return final_timestamps
        
    except OpenAIServiceError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_topic_timestamps: {str(e)}")
        raise OpenAIServiceError(f"Failed to generate timestamps: {str(e)}")


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