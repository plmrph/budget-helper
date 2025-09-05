import { writable, derived } from 'svelte/store';

/**
 * Create a form store for managing form state, validation, and submission
 * @param {Object} initialValues - Initial form values
 * @param {Object} validationRules - Validation rules for each field
 * @returns {Object} Form store with methods and reactive values
 */
export function createFormStore(initialValues = {}, validationRules = {}) {
	// Core form state
	const values = writable({ ...initialValues });
	const errors = writable({});
	const touched = writable({});
	const submitting = writable(false);
	const submitted = writable(false);

	// Derived states
	const isValid = derived([errors], ([$errors]) => {
		return Object.keys($errors).length === 0;
	});

	const isDirty = derived([values], ([$values]) => {
		return JSON.stringify($values) !== JSON.stringify(initialValues);
	});

	// Validation function
	function validateField(fieldName, value) {
		const rule = validationRules[fieldName];
		if (!rule) return null;

		if (typeof rule === 'function') {
			return rule(value);
		}

		if (typeof rule === 'object') {
			// Required validation
			if (rule.required && (!value || (typeof value === 'string' && value.trim() === ''))) {
				return rule.requiredMessage || `${fieldName} is required`;
			}

			// Min length validation
			if (rule.minLength && value && value.length < rule.minLength) {
				return rule.minLengthMessage || `${fieldName} must be at least ${rule.minLength} characters`;
			}

			// Max length validation
			if (rule.maxLength && value && value.length > rule.maxLength) {
				return rule.maxLengthMessage || `${fieldName} must be no more than ${rule.maxLength} characters`;
			}

			// Pattern validation
			if (rule.pattern && value && !rule.pattern.test(value)) {
				return rule.patternMessage || `${fieldName} format is invalid`;
			}

			// Custom validation function
			if (rule.validate && typeof rule.validate === 'function') {
				return rule.validate(value);
			}
		}

		return null;
	}

	// Methods
	const methods = {
		// Set field value
		setValue(fieldName, value) {
			values.update(v => ({ ...v, [fieldName]: value }));
			
			// Mark field as touched
			touched.update(t => ({ ...t, [fieldName]: true }));
			
			// Validate field
			const error = validateField(fieldName, value);
			errors.update(e => {
				const newErrors = { ...e };
				if (error) {
					newErrors[fieldName] = error;
				} else {
					delete newErrors[fieldName];
				}
				return newErrors;
			});
		},

		// Set multiple values
		setValues(newValues) {
			values.update(v => ({ ...v, ...newValues }));
			
			// Mark all fields as touched
			touched.update(t => {
				const newTouched = { ...t };
				Object.keys(newValues).forEach(key => {
					newTouched[key] = true;
				});
				return newTouched;
			});
			
			// Validate all changed fields
			errors.update(e => {
				const newErrors = { ...e };
				Object.entries(newValues).forEach(([fieldName, value]) => {
					const error = validateField(fieldName, value);
					if (error) {
						newErrors[fieldName] = error;
					} else {
						delete newErrors[fieldName];
					}
				});
				return newErrors;
			});
		},

		// Set field error
		setError(fieldName, error) {
			errors.update(e => ({ ...e, [fieldName]: error }));
		},

		// Clear field error
		clearError(fieldName) {
			errors.update(e => {
				const newErrors = { ...e };
				delete newErrors[fieldName];
				return newErrors;
			});
		},

		// Clear all errors
		clearErrors() {
			errors.set({});
		},

		// Mark field as touched
		setTouched(fieldName, isTouched = true) {
			touched.update(t => ({ ...t, [fieldName]: isTouched }));
		},

		// Validate all fields
		validate() {
			let hasErrors = false;
			const newErrors = {};
			
			values.subscribe(v => {
				Object.entries(v).forEach(([fieldName, value]) => {
					const error = validateField(fieldName, value);
					if (error) {
						newErrors[fieldName] = error;
						hasErrors = true;
					}
				});
			})();
			
			errors.set(newErrors);
			
			// Mark all fields as touched
			values.subscribe(v => {
				const allTouched = {};
				Object.keys(v).forEach(key => {
					allTouched[key] = true;
				});
				touched.set(allTouched);
			})();
			
			return !hasErrors;
		},

		// Submit form
		async submit(submitHandler) {
			submitting.set(true);
			submitted.set(true);
			
			try {
				// Validate before submitting
				const isFormValid = methods.validate();
				if (!isFormValid) {
					return { success: false, errors: 'Validation failed' };
				}
				
				// Get current values
				let currentValues;
				values.subscribe(v => { currentValues = v; })();
				
				// Call submit handler
				const result = await submitHandler(currentValues);
				return result;
			} catch (error) {
				console.error('Form submission error:', error);
				return { success: false, error: error.message };
			} finally {
				submitting.set(false);
			}
		},

		// Reset form
		reset() {
			values.set({ ...initialValues });
			errors.set({});
			touched.set({});
			submitting.set(false);
			submitted.set(false);
		},

		// Get field props for binding to form components
		getFieldProps(fieldName) {
			return {
				get value() {
					let currentValue;
					values.subscribe(v => { currentValue = v[fieldName]; })();
					return currentValue;
				},
				set value(newValue) {
					methods.setValue(fieldName, newValue);
				},
				get error() {
					let currentError;
					errors.subscribe(e => { currentError = e[fieldName]; })();
					return currentError;
				},
				get touched() {
					let isTouched;
					touched.subscribe(t => { isTouched = t[fieldName]; })();
					return isTouched;
				}
			};
		}
	};

	return {
		// Stores
		values,
		errors,
		touched,
		submitting,
		submitted,
		isValid,
		isDirty,
		
		// Methods
		...methods
	};
}

/**
 * Create a simple field store for individual field management
 * @param {*} initialValue - Initial field value
 * @param {Function|Object} validation - Validation rule
 * @returns {Object} Field store with methods and reactive values
 */
export function createFieldStore(initialValue = '', validation = null) {
	const value = writable(initialValue);
	const error = writable('');
	const touched = writable(false);

	function validate(val) {
		if (!validation) return true;
		
		let errorMessage = '';
		if (typeof validation === 'function') {
			errorMessage = validation(val) || '';
		} else if (typeof validation === 'object') {
			// Similar validation logic as in form store
			if (validation.required && (!val || (typeof val === 'string' && val.trim() === ''))) {
				errorMessage = validation.requiredMessage || 'Field is required';
			}
			// Add other validation rules as needed
		}
		
		error.set(errorMessage);
		return !errorMessage;
	}

	return {
		value,
		error,
		touched,
		
		setValue(newValue) {
			value.set(newValue);
			touched.set(true);
			validate(newValue);
		},
		
		setError(errorMessage) {
			error.set(errorMessage);
		},
		
		clearError() {
			error.set('');
		},
		
		validate(val) {
			const currentValue = val !== undefined ? val : value;
			return validate(currentValue);
		},
		
		reset() {
			value.set(initialValue);
			error.set('');
			touched.set(false);
		}
	};
}