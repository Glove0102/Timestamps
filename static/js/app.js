// SRT Timestamp Generator - Client-side JavaScript

document.addEventListener('DOMContentLoaded', function() {
    initializeFileUpload();
    initializeFormSubmission();
});

/**
 * Initialize file upload functionality with drag and drop
 */
function initializeFileUpload() {
    const fileUploadArea = document.getElementById('fileUploadArea');
    const fileInput = document.getElementById('srt_file');
    const fileUploadContent = document.querySelector('.file-upload-content');
    const fileSelected = document.getElementById('fileSelected');
    const selectedFileName = document.getElementById('selectedFileName');

    if (!fileUploadArea || !fileInput) return;

    // Drag and drop events
    fileUploadArea.addEventListener('dragover', handleDragOver);
    fileUploadArea.addEventListener('dragleave', handleDragLeave);
    fileUploadArea.addEventListener('drop', handleDrop);
    
    // Add click handler to the choose file button specifically
    const chooseFileBtn = document.getElementById('chooseFileBtn');
    if (chooseFileBtn) {
        chooseFileBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            fileInput.click();
        });
    }

    // File input change event
    fileInput.addEventListener('change', handleFileSelect);

    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        fileUploadArea.classList.add('dragover');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        fileUploadArea.classList.remove('dragover');
    }

    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        fileUploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (validateFile(file)) {
                updateFileInput(file);
                showSelectedFile(file.name);
            }
        }
    }

    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file && validateFile(file)) {
            showSelectedFile(file.name);
        }
    }

    function validateFile(file) {
        const allowedTypes = ['.srt'];
        const fileName = file.name.toLowerCase();
        const isValidType = allowedTypes.some(type => fileName.endsWith(type));

        if (!isValidType) {
            showAlert('Please select a valid SRT file.', 'error');
            return false;
        }

        const maxSize = 16 * 1024 * 1024; // 16MB
        if (file.size > maxSize) {
            showAlert('File is too large. Please select a file smaller than 16MB.', 'error');
            return false;
        }

        return true;
    }

    function updateFileInput(file) {
        try {
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;
            
            // Trigger change event to ensure form validation works
            const changeEvent = new Event('change', { bubbles: true });
            fileInput.dispatchEvent(changeEvent);
        } catch (error) {
            console.log('DataTransfer not supported, using fallback');
            // Create a new file input with the dropped file
            const newInput = document.createElement('input');
            newInput.type = 'file';
            newInput.name = 'srt_file';
            newInput.accept = '.srt';
            newInput.style.display = 'none';
            
            // Replace the old input
            fileInput.parentNode.replaceChild(newInput, fileInput);
            fileInput = newInput;
        }
    }

    function showSelectedFile(fileName) {
        selectedFileName.textContent = fileName;
        fileUploadContent.classList.add('d-none');
        fileSelected.classList.remove('d-none');
    }
}

/**
 * Clear selected file
 */
function clearFile() {
    const fileInput = document.getElementById('srt_file');
    const fileUploadContent = document.querySelector('.file-upload-content');
    const fileSelected = document.getElementById('fileSelected');

    if (fileInput) fileInput.value = '';
    if (fileUploadContent) fileUploadContent.classList.remove('d-none');
    if (fileSelected) fileSelected.classList.add('d-none');
}

/**
 * Initialize form submission with loading state
 */
function initializeFormSubmission() {
    const form = document.getElementById('uploadForm');
    const submitBtn = document.getElementById('submitBtn');

    if (!form || !submitBtn) return;

    form.addEventListener('submit', function(e) {
        const fileInput = document.getElementById('srt_file');
        
        if (!fileInput || !fileInput.files.length) {
            e.preventDefault();
            showAlert('Please select an SRT file to upload.', 'error');
            return;
        }

        // Show loading state
        showLoadingState(true);
        
        // Disable form to prevent double submission
        const formElements = form.querySelectorAll('input, textarea, button');
        formElements.forEach(element => element.disabled = true);
    });
}

/**
 * Show/hide loading state
 */
function showLoadingState(isLoading) {
    const submitBtn = document.getElementById('submitBtn');
    const form = document.getElementById('uploadForm');
    
    if (!submitBtn || !form) return;

    if (isLoading) {
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
        form.classList.add('loading');
    } else {
        submitBtn.innerHTML = '<i class="fas fa-magic me-2"></i>Generate Timestamps';
        form.classList.remove('loading');
    }
}

/**
 * Copy timestamps to clipboard
 */
async function copyTimestamps() {
    const timestampsContainer = document.getElementById('timestampsContent');
    if (!timestampsContainer) return;

    try {
        // Extract text from timestamp items
        const timestampItems = timestampsContainer.querySelectorAll('.timestamp-item');
        const timestamps = Array.from(timestampItems).map(item => {
            // Remove the clock icon and get just the text
            return item.textContent.trim();
        });
        
        const timestampText = timestamps.join('\n');

        // Use the modern Clipboard API
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(timestampText);
        } else {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = timestampText;
            textArea.style.position = 'fixed';
            textArea.style.left = '-999999px';
            textArea.style.top = '-999999px';
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            document.execCommand('copy');
            textArea.remove();
        }

        // Show success toast
        showCopyToast();
        
    } catch (error) {
        console.error('Failed to copy timestamps:', error);
        showAlert('Failed to copy timestamps. Please try selecting and copying manually.', 'error');
    }
}

/**
 * Show copy success toast
 */
function showCopyToast() {
    const toast = document.getElementById('copyToast');
    if (toast) {
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
    }
}

/**
 * Show alert message
 */
function showAlert(message, type = 'info') {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
    alertDiv.setAttribute('role', 'alert');
    
    const icon = type === 'error' ? 'exclamation-triangle' : 
                 type === 'success' ? 'check-circle' : 'info-circle';
    
    alertDiv.innerHTML = `
        <i class="fas fa-${icon} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;

    // Insert at the top of the main container
    const main = document.querySelector('main.container');
    if (main) {
        main.insertBefore(alertDiv, main.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                const bsAlert = new bootstrap.Alert(alertDiv);
                bsAlert.close();
            }
        }, 5000);
    }
}

/**
 * Auto-dismiss alerts after 5 seconds
 */
document.addEventListener('DOMContentLoaded', function() {
    const alerts = document.querySelectorAll('.alert:not(.alert-info)');
    alerts.forEach(alert => {
        setTimeout(() => {
            if (alert.parentNode) {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }
        }, 5000);
    });
});
