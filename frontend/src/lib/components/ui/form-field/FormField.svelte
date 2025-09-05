<script>
	import { cn } from "$lib/utils.js";
	import { Input } from "../input/index.js";
	import { Label } from "../label/index.js";

	let {
		value = $bindable(""),
		label = "",
		placeholder = "",
		error = "",
		required = false,
		disabled = false,
		type = "text",
		class: className = "",
		inputClass = "",
		labelClass = "",
		errorClass = "",
		onValidate = null,
		onSubmit = null,
		inline = false,
		editing = $bindable(false),
		...restProps
	} = $props();

	let inputElement;
	let originalValue = $state(value);
	let isValid = $state(true);

	// Validation function
	function validate() {
		if (onValidate) {
			const validationResult = onValidate(value);
			if (typeof validationResult === 'string') {
				error = validationResult;
				isValid = false;
			} else {
				error = "";
				isValid = validationResult !== false;
			}
		} else {
			// Basic required validation
			if (required && (!value || value.trim() === "")) {
				error = `${label || 'Field'} is required`;
				isValid = false;
			} else {
				error = "";
				isValid = true;
			}
		}
		return isValid;
	}

	// Handle input changes
	function handleInput(event) {
		value = event.target.value;
		if (error) {
			validate(); // Clear error on input if there was one
		}
	}

	// Handle blur (validation)
	function handleBlur() {
		validate();
		if (inline && editing) {
			handleSubmit();
		}
	}

	// Handle key events
	function handleKeydown(event) {
		if (event.key === 'Enter') {
			event.preventDefault();
			handleSubmit();
		} else if (event.key === 'Escape') {
			event.preventDefault();
			handleCancel();
		}
	}

	// Handle form submission
	function handleSubmit() {
		if (validate()) {
			if (onSubmit) {
				onSubmit(value);
			}
			if (inline) {
				editing = false;
			}
		}
	}

	// Handle cancel (for inline editing)
	function handleCancel() {
		value = originalValue;
		error = "";
		if (inline) {
			editing = false;
		}
	}

	// Start inline editing
	function startEditing(event) {
		if (inline && !disabled) {
			originalValue = value;
			editing = true;
			// Focus input after DOM update - use the specific container
			setTimeout(() => {
				const container = event.target.closest('td') || event.target.closest('div');
				const input = container?.querySelector('input[data-slot="input"]');
				if (input) {
					input.focus();
					input.select();
				}
			}, 0);
		}
	}

	// Update original value when value changes externally
	$effect(() => {
		if (!editing) {
			originalValue = value;
		}
	});
</script>

<div class={cn("space-y-2", className)} {...restProps}>
	{#if label && !inline}
		<Label class={cn("block", labelClass)}>
			{label}
			{#if required}
				<span class="text-red-500 ml-1">*</span>
			{/if}
		</Label>
	{/if}

	{#if inline}
		{#if editing}
			<Input
				bind:value
				{type}
				{placeholder}
				{disabled}
				class={cn(
					"h-8 text-sm",
					error ? "border-red-500 focus:border-red-500 focus:ring-red-500" : "",
					inputClass
				)}
				oninput={handleInput}
				onblur={handleBlur}
				onkeydown={handleKeydown}
			/>
		{:else}
			<button
				type="button"
				class={cn(
					"text-left w-full px-2 py-1 text-sm rounded hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500",
					disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer",
					!value ? "text-gray-400 italic" : ""
				)}
				onclick={startEditing}
				{disabled}
			>
				{value || placeholder || "Click to edit"}
			</button>
		{/if}
	{:else}
		<Input
			bind:value
			{type}
			{placeholder}
			{disabled}
			class={cn(
				error ? "border-red-500 focus:border-red-500 focus:ring-red-500" : "",
				inputClass
			)}
			oninput={handleInput}
			onblur={handleBlur}
			onkeydown={handleKeydown}
		/>
	{/if}

	{#if error}
		<p class={cn("text-sm text-red-600", errorClass)}>
			{error}
		</p>
	{/if}
</div>