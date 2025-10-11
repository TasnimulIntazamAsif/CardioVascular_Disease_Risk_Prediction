// JavaScript for CVD Risk Assessment System

document.addEventListener('DOMContentLoaded', function() {
    // Initialize form
    initializeForm();
    
    // Set up event listeners
    setupEventListeners();
    
    // Auto-calculate BMI
    setupBMICalculation();
});

function initializeForm() {
    // Add smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

function setupEventListeners() {
    // Form submission
    const form = document.getElementById('assessmentForm');
    if (form) {
        form.addEventListener('submit', handleFormSubmission);
    }
    
    // Real-time validation
    const inputs = document.querySelectorAll('.form-control, .form-select');
    inputs.forEach(input => {
        input.addEventListener('blur', validateField);
        input.addEventListener('input', clearValidation);
    });
    
    // Height conversion
    const heightM = document.getElementById('height_m');
    const heightCm = document.getElementById('height_cm');
    
    if (heightM && heightCm) {
        heightM.addEventListener('input', function() {
            if (this.value) {
                heightCm.value = Math.round(this.value * 100);
            }
        });
        
        heightCm.addEventListener('input', function() {
            if (this.value) {
                heightM.value = (this.value / 100).toFixed(2);
            }
        });
    }
}

function setupBMICalculation() {
    const weightInput = document.getElementById('weight');
    const heightMInput = document.getElementById('height_m');
    const bmiInput = document.getElementById('bmi');
    
    function calculateBMI() {
        const weight = parseFloat(weightInput.value);
        const height = parseFloat(heightMInput.value);
        
        if (weight && height && height > 0) {
            const bmi = weight / (height * height);
            bmiInput.value = bmi.toFixed(1);
            
            // Add BMI category indicator
            updateBMICategory(bmi);
        } else {
            bmiInput.value = '';
        }
    }
    
    if (weightInput) weightInput.addEventListener('input', calculateBMI);
    if (heightMInput) heightMInput.addEventListener('input', calculateBMI);
}

function updateBMICategory(bmi) {
    const bmiInput = document.getElementById('bmi');
    if (!bmiInput) return;
    
    // Remove existing category classes
    bmiInput.classList.remove('text-success', 'text-warning', 'text-danger');
    
    if (bmi < 18.5) {
        bmiInput.classList.add('text-info');
        bmiInput.title = 'Underweight';
    } else if (bmi < 25) {
        bmiInput.classList.add('text-success');
        bmiInput.title = 'Normal weight';
    } else if (bmi < 30) {
        bmiInput.classList.add('text-warning');
        bmiInput.title = 'Overweight';
    } else {
        bmiInput.classList.add('text-danger');
        bmiInput.title = 'Obese';
    }
}

function validateField(event) {
    const field = event.target;
    const value = field.value.trim();
    
    // Remove existing validation classes
    field.classList.remove('is-valid', 'is-invalid');
    
    // Check if field is required
    if (field.hasAttribute('required') && !value) {
        field.classList.add('is-invalid');
        return false;
    }
    
    // Specific validations
    if (field.type === 'number') {
        const numValue = parseFloat(value);
        const min = parseFloat(field.getAttribute('min'));
        const max = parseFloat(field.getAttribute('max'));
        
        if (value && (isNaN(numValue) || (min && numValue < min) || (max && numValue > max))) {
            field.classList.add('is-invalid');
            return false;
        }
    }
    
    // Age validation
    if (field.name === 'Age') {
        const age = parseInt(value);
        if (age < 18 || age > 100) {
            field.classList.add('is-invalid');
            return false;
        }
    }
    
    // Blood pressure validation
    if (field.name === 'Systolic BP') {
        const systolic = parseInt(value);
        if (systolic < 70 || systolic > 250) {
            field.classList.add('is-invalid');
            return false;
        }
    }
    
    if (field.name === 'Diastolic BP') {
        const diastolic = parseInt(value);
        if (diastolic < 40 || diastolic > 150) {
            field.classList.add('is-invalid');
            return false;
        }
    }
    
    // If we get here, field is valid
    if (value) {
        field.classList.add('is-valid');
    }
    
    return true;
}

function clearValidation(event) {
    const field = event.target;
    field.classList.remove('is-valid', 'is-invalid');
}

function validateForm() {
    const form = document.getElementById('assessmentForm');
    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!validateField({ target: field })) {
            isValid = false;
        }
    });
    
    return isValid;
}

async function handleFormSubmission(event) {
    event.preventDefault();
    
    // Validate form
    if (!validateForm()) {
        showAlert('Please fill in all required fields correctly.', 'danger');
        return;
    }
    
    // Show loading modal
    const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
    loadingModal.show();
    
    try {
        // Collect form data
        const formData = new FormData(event.target);
        
        // Send prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        // Hide loading modal
        loadingModal.hide();
        
        if (result.error) {
            showAlert(result.error, 'danger');
        } else {
            displayResults(result);
        }
        
    } catch (error) {
        loadingModal.hide();
        showAlert('An error occurred while processing your request. Please try again.', 'danger');
        console.error('Error:', error);
    }
}

function displayResults(result) {
    const resultsSection = document.getElementById('results');
    const riskResult = document.getElementById('riskResult');
    const suggestions = document.getElementById('suggestions');
    
    // Create risk level display
    const riskClass = `risk-${result.risk_level.toLowerCase()}`;
    const riskIcon = getRiskIcon(result.risk_level);
    
    riskResult.innerHTML = `
        <div class="risk-card ${riskClass}">
            <div class="risk-icon">${riskIcon}</div>
            <div class="risk-title">${result.suggestions.title}</div>
            <div class="risk-confidence">
                <i class="fas fa-chart-line me-2"></i>
                Confidence: ${result.confidence}%
            </div>
        </div>
    `;
    
    // Create suggestions display
    suggestions.innerHTML = `
        <div class="suggestions-card">
            <h4 class="suggestions-title">
                <i class="fas fa-lightbulb me-2"></i>
                Recommendations
            </h4>
            <ul class="suggestions-list">
                ${result.suggestions.suggestions.map(suggestion => 
                    `<li>${suggestion}</li>`
                ).join('')}
            </ul>
        </div>
    `;
    
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    // Add pulse animation to risk card
    const riskCard = riskResult.querySelector('.risk-card');
    riskCard.classList.add('pulse');
}

function getRiskIcon(riskLevel) {
    switch (riskLevel) {
        case 'LOW':
            return '<i class="fas fa-check-circle"></i>';
        case 'INTERMEDIARY':
            return '<i class="fas fa-exclamation-triangle"></i>';
        case 'HIGH':
            return '<i class="fas fa-exclamation-circle"></i>';
        default:
            return '<i class="fas fa-question-circle"></i>';
    }
}

function showAlert(message, type) {
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.alert');
    existingAlerts.forEach(alert => alert.remove());
    
    // Create new alert
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at top of form
    const form = document.getElementById('assessmentForm');
    form.insertBefore(alertDiv, form.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

function resetForm() {
    // Reset form
    document.getElementById('assessmentForm').reset();
    
    // Clear validation classes
    const inputs = document.querySelectorAll('.form-control, .form-select');
    inputs.forEach(input => {
        input.classList.remove('is-valid', 'is-invalid');
    });
    
    // Hide results
    document.getElementById('results').style.display = 'none';
    
    // Scroll to top of form
    document.getElementById('assessment').scrollIntoView({ behavior: 'smooth' });
    
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.alert');
    existingAlerts.forEach(alert => alert.remove());
}

// Utility functions
function formatNumber(value, decimals = 1) {
    return parseFloat(value).toFixed(decimals);
}

function validateBloodPressure(systolic, diastolic) {
    if (systolic <= diastolic) {
        return false;
    }
    return true;
}

// Add blood pressure validation
document.addEventListener('DOMContentLoaded', function() {
    const systolicInput = document.getElementById('systolic');
    const diastolicInput = document.getElementById('diastolic');
    
    if (systolicInput && diastolicInput) {
        function validateBP() {
            const systolic = parseInt(systolicInput.value);
            const diastolic = parseInt(diastolicInput.value);
            
            if (systolic && diastolic) {
                if (!validateBloodPressure(systolic, diastolic)) {
                    systolicInput.classList.add('is-invalid');
                    diastolicInput.classList.add('is-invalid');
                    return false;
                } else {
                    systolicInput.classList.remove('is-invalid');
                    diastolicInput.classList.remove('is-invalid');
                    return true;
                }
            }
        }
        
        systolicInput.addEventListener('blur', validateBP);
        diastolicInput.addEventListener('blur', validateBP);
    }
});

// Add tooltips for better user experience
document.addEventListener('DOMContentLoaded', function() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

// Smooth animations for form sections
function animateFormSections() {
    const sections = document.querySelectorAll('.row.mb-4');
    sections.forEach((section, index) => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            section.style.opacity = '1';
            section.style.transform = 'translateY(0)';
        }, index * 100);
    });
}

// Initialize animations when page loads
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(animateFormSections, 500);
});
