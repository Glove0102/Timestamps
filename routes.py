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
        # Validate file upload
        if 'srt_file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        file = request.files['srt_file']
        context = request.form.get('context', '').strip()
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        if not file or not allowed_file(file.filename):
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
            
            # Generate timestamps using OpenAI
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

@app.route('/result/<int:request_id>')
def get_result(request_id):
    """Retrieve generated timestamps by request ID"""
    try:
        timestamp_request = TimestampRequest.query.get_or_404(request_id)
        
        if timestamp_request.status == 'completed' and timestamp_request.generated_timestamps:
            timestamps = timestamp_request.generated_timestamps.split('\n')
            return render_template('index.html',
                                 timestamps=timestamps,
                                 filename=timestamp_request.filename,
                                 request_id=request_id)
        elif timestamp_request.status == 'failed':
            flash(f'Request failed: {timestamp_request.error_message}', 'error')
            return redirect(url_for('index'))
        else:
            flash('Request is still processing. Please try again in a moment.', 'info')
            return redirect(url_for('index'))
    
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
