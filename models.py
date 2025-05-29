from app import db
from datetime import datetime
from sqlalchemy import func

class TimestampRequest(db.Model):
    """Model for storing timestamp generation requests and results"""
    __tablename__ = 'timestamp_requests'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    context_provided = db.Column(db.Text, nullable=True)
    generated_timestamps = db.Column(db.Text, nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default='processing')  # processing, completed, failed
    created_at = db.Column(db.DateTime, default=func.now())
    
    def __repr__(self):
        return f'<TimestampRequest {self.id}: {self.filename}>'
    
    def to_dict(self):
        """Convert model to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'filename': self.filename,
            'context_provided': self.context_provided,
            'generated_timestamps': self.generated_timestamps,
            'error_message': self.error_message,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
