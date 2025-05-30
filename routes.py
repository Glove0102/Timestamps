import os
import logging
from flask import render_template, request, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from app import app, db
from models import TimestampRequest
from srt_parser import parse_srt_file, SRTParseError
from openai_service import generate_topic_timestamps, OpenAIServiceError

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'srt'}

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main upload page"""
    return render_template('index.html')

@app.route('/generate-timestamps', methods=['POST'])
def generate_timestamps():
    """Handle SRT file upload and generate topic-based timestamps"""
    try:
        # Debug logging
        logger.debug(f"Request files: {list(request.files.keys())}")
        logger.debug(f"Request form: {dict(request.form)}")
        
        # Validate file upload
        if 'srt_file' not in request.files:
            logger.warning("No 'srt_file' key in request.files")
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        file = request.files['srt_file']
        context = request.form.get('context', '').strip()
        
        logger.debug(f"File object: {file}, filename: '{file.filename}'")
        
        if not file or file.filename == '':
            logger.warning(f"No file selected or empty filename: {file}")
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        # Check file extension
        if not file.filename or not allowed_file(file.filename):
            flash('Please upload a valid SRT file', 'error')
            return redirect(url_for('index'))
        
        # Secure filename
        filename = secure_filename(file.filename)
        
        # Create database record
        timestamp_request = TimestampRequest(
            filename=filename,
            context_provided=context if context else None,
            status='processing'
        )
        db.session.add(timestamp_request)
        db.session.commit()
        
        try:
            # Read and parse SRT file
            srt_content = file.read().decode('utf-8')
            logger.debug(f"Read SRT file: {len(srt_content)} characters")
            
            srt_entries = parse_srt_file(srt_content)
            logger.debug(f"Parsed {len(srt_entries)} SRT entries")
            
            if not srt_entries:
                raise SRTParseError("No valid SRT entries found in file")
            
            # For large files, redirect to processing page immediately
            # Calculate estimated processing time
            first_time = float(srt_entries[0]['start'].split(':')[0]) * 3600 + float(srt_entries[0]['start'].split(':')[1]) * 60 + float(srt_entries[0]['start'].split(':')[2].replace(',', '.'))
            last_time = float(srt_entries[-1]['end'].split(':')[0]) * 3600 + float(srt_entries[-1]['end'].split(':')[1]) * 60 + float(srt_entries[-1]['end'].split(':')[2].replace(',', '.'))
            video_duration_hours = (last_time - first_time) / 3600
            
            # If video is longer than 45 minutes, process in background
            if video_duration_hours > 0.75:  # 45 minutes
                # Store SRT data in database for background processing
                timestamp_request.context_provided = f"SRT_DATA:{srt_content}"
                if context:
                    timestamp_request.context_provided += f"|CONTEXT:{context}"
                db.session.commit()
                
                flash(f'Large file detected ({video_duration_hours:.1f} hours). Processing in background...', 'info')
                return redirect(url_for('get_result', request_id=timestamp_request.id))
            
            # For smaller files, process immediately
            try:
                timestamps = generate_topic_timestamps(srt_entries, context)
                logger.debug(f"Generated {len(timestamps)} topic timestamps")
                
                # Update database record with results
                timestamp_request.generated_timestamps = '\n'.join(timestamps)
                timestamp_request.status = 'completed'
                db.session.commit()
                
                flash('Timestamps generated successfully!', 'success')
                return render_template('index.html', 
                                     timestamps=timestamps,
                                     filename=filename,
                                     request_id=timestamp_request.id)
            except Exception as e:
                logger.error(f"Error processing small file: {str(e)}")
                # Fall back to background processing for small files that fail
                timestamp_request.context_provided = f"SRT_DATA:{srt_content}"
                if context:
                    timestamp_request.context_provided += f"|CONTEXT:{context}"
                db.session.commit()
                
                flash('Processing in background due to complexity...', 'info')
                return redirect(url_for('get_result', request_id=timestamp_request.id))
        
        except (SRTParseError, OpenAIServiceError) as e:
            logger.error(f"Error processing file: {str(e)}")
            timestamp_request.error_message = str(e)
            timestamp_request.status = 'failed'
            db.session.commit()
            
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(url_for('index'))
        
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            timestamp_request.error_message = f"Unexpected error: {str(e)}"
            timestamp_request.status = 'failed'
            db.session.commit()
            
            flash('An unexpected error occurred. Please try again.', 'error')
            return redirect(url_for('index'))
    
    except Exception as e:
        logger.error(f"Critical error in generate_timestamps: {str(e)}")
        flash('A critical error occurred. Please try again.', 'error')
        return redirect(url_for('index'))

def process_background_request(request_id):
    """Process a request in the background"""
    try:
        timestamp_request = TimestampRequest.query.get(request_id)
        if not timestamp_request or timestamp_request.status != 'processing':
            return
        
        # Extract SRT data and context from stored data
        context_data = timestamp_request.context_provided or ""
        if context_data.startswith("SRT_DATA:"):
            parts = context_data.split("|CONTEXT:")
            srt_content = parts[0][9:]  # Remove "SRT_DATA:" prefix
            context = parts[1] if len(parts) > 1 else None
            
            # Parse SRT and generate timestamps
            srt_entries = parse_srt_file(srt_content)
            timestamps = generate_topic_timestamps(srt_entries, context)
            
            # Update database
            timestamp_request.generated_timestamps = '\n'.join(timestamps)
            timestamp_request.status = 'completed'
            db.session.commit()
            
            logger.info(f"Background processing completed for request {request_id}")
            
    except Exception as e:
        logger.error(f"Background processing failed for request {request_id}: {str(e)}")
        timestamp_request.error_message = str(e)
        timestamp_request.status = 'failed'
        db.session.commit()


@app.route('/result/<int:request_id>')
def get_result(request_id):
    """Retrieve generated timestamps by request ID"""
    try:
        timestamp_request = TimestampRequest.query.get_or_404(request_id)
        
        # If still processing and has SRT data, try processing now
        if (timestamp_request.status == 'processing' and 
            timestamp_request.context_provided and 
            timestamp_request.context_provided.startswith("SRT_DATA:")):
            
            try:
                process_background_request(request_id)
                # Refresh the request object
                timestamp_request = TimestampRequest.query.get(request_id)
            except Exception as e:
                logger.error(f"Error in background processing: {str(e)}")
        
        if timestamp_request.status == 'completed' and timestamp_request.generated_timestamps:
            timestamps = timestamp_request.generated_timestamps.split('\n')
            return render_template('index.html',
                                 timestamps=timestamps,
                                 filename=timestamp_request.filename,
                                 request_id=request_id)
        elif timestamp_request.status == 'failed':
            flash(f'Processing failed: {timestamp_request.error_message}', 'error')
            return redirect(url_for('index'))
        else:
            # Create a processing status page
            return render_template('processing.html',
                                 filename=timestamp_request.filename,
                                 request_id=request_id)
    
    except Exception as e:
        logger.error(f"Error retrieving result: {str(e)}")
        flash('Error retrieving results. Please try again.', 'error')
        return redirect(url_for('index'))

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large. Please upload a file smaller than 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    flash('The requested page was not found.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(e)}")
    flash('An internal server error occurred. Please try again.', 'error')
    return redirect(url_for('index'))
