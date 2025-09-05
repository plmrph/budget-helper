<script>
	import { cn } from "$lib/utils.js";
	import { Input } from "../input/index.js";
	import { Button } from "../button/index.js";
	import { Popover } from "../popover/index.js";

	let {
		value = $bindable(""),
		options = [],
		placeholder = "Select an option...",
		searchPlaceholder = "Search...",
		disabled = false,
		searchable = true,
		clearable = false,
		class: className = "",
		buttonClass = "",
		contentClass = "",
		optionClass = "",
		onSelect = null,
		inline = false,
		editing = $bindable(false),
		displayValue = null, // Function to format display value
		...restProps
	} = $props();

	let open = $state(false);
	let searchQuery = $state("");
	let originalValue = $state(value);
	let dropdownPosition = $state('bottom'); // 'bottom' or 'top'

	// Filter options based on search query
	const filteredOptions = $derived(searchQuery
		? options.filter(option => {
				const label = typeof option === 'string' ? option : option.label || option.name || String(option.value);
				return label.toLowerCase().includes(searchQuery.toLowerCase());
			})
		: options);

	// Get display text for selected value
	function getDisplayText(val) {
		if (!val) return placeholder;
		
		if (displayValue) {
			return displayValue(val);
		}
		
		const option = options.find(opt => {
			const optValue = typeof opt === 'string' ? opt : opt.value;
			return optValue === val;
		});
		
		if (option) {
			return typeof option === 'string' ? option : option.label || option.name || String(option.value);
		}
		
		return String(val);
	}

	// Handle option selection
	function selectOption(option) {
		const optValue = typeof option === 'string' ? option : option.value;
		value = optValue;
		open = false;
		searchQuery = "";
		
		if (onSelect) {
			onSelect(optValue, option);
		}
		
		if (inline) {
			editing = false;
		}
	}

	// Handle clear selection
	function clearSelection() {
		value = "";
		open = false;
		searchQuery = "";
		
		if (onSelect) {
			onSelect("", null);
		}
		
		if (inline) {
			editing = false;
		}
	}

	// Start inline editing
	function startEditing(event) {
		if (inline && !disabled) {
			originalValue = value;
			editing = true;
			open = true;
			// Calculate dropdown position
			calculateDropdownPosition(event.target);
		}
	}

	// Calculate whether dropdown should appear above or below
	function calculateDropdownPosition(element) {
		const rect = element.getBoundingClientRect();
		const viewportHeight = window.innerHeight;
		const spaceBelow = viewportHeight - rect.bottom;
		const spaceAbove = rect.top;
		
		// If there's less than 200px below but more than 200px above, show above
		dropdownPosition = (spaceBelow < 200 && spaceAbove > 200) ? 'top' : 'bottom';
	}

	// Handle cancel (for inline editing)
	function handleCancel() {
		value = originalValue;
		open = false;
		searchQuery = "";
		if (inline) {
			editing = false;
		}
	}

	// Handle key events
	function handleKeydown(event) {
		if (event.key === 'Escape') {
			event.preventDefault();
			handleCancel();
		}
	}

	// Update original value when value changes externally
	$effect(() => {
		if (!editing) {
			originalValue = value;
		}
	});
</script>

<div class={cn("relative", className)} {...restProps}>
	{#if inline}
		{#if editing}
			<Popover bind:open>
				{#snippet children({ trigger, content })}
					<Button
						{...trigger}
						variant="outline"
						class={cn(
							"w-full justify-between h-8 text-sm",
							buttonClass
						)}
						{disabled}
					>
						<span class="truncate">
							{getDisplayText(value)}
						</span>
						<svg class="h-4 w-4 opacity-50" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
							<path d="m6 9 6 6 6-6"/>
						</svg>
					</Button>
					
					{#if open}
						<div
							{...content}
							class={cn(
								"w-full min-w-[200px] max-h-60 overflow-hidden",
								dropdownPosition === 'top' ? 'mb-1 bottom-full' : 'mt-1 top-full',
								content.class,
								contentClass
							)}
							onkeydown={handleKeydown}
						>
							{#if searchable}
								<div class="p-2 border-b">
									<Input
										bind:value={searchQuery}
										placeholder={searchPlaceholder}
										class="h-8 text-sm"
									/>
								</div>
							{/if}
							
							<div class="max-h-60 overflow-y-auto">
								{#if clearable && value}
									<button
										type="button"
										class={cn(
											"w-full px-3 py-2 text-left text-sm hover:bg-gray-100 focus:bg-gray-100 focus:outline-none border-b",
											optionClass
										)}
										onclick={clearSelection}
									>
										<span class="text-gray-500 italic">Clear selection</span>
									</button>
								{/if}
								
								{#each filteredOptions as option}
									{@const optValue = typeof option === 'string' ? option : option.value}
									{@const optLabel = typeof option === 'string' ? option : option.label || option.name || String(option.value)}
									<button
										type="button"
										class={cn(
											"w-full px-3 py-2 text-left text-sm hover:bg-gray-100 focus:bg-gray-100 focus:outline-none",
											value === optValue ? "bg-blue-50 text-blue-700" : "",
											optionClass
										)}
										onclick={() => selectOption(option)}
									>
										{optLabel}
									</button>
								{:else}
									<div class="px-3 py-2 text-sm text-gray-500 italic">
										No options found
									</div>
								{/each}
							</div>
						</div>
					{/if}
				{/snippet}
			</Popover>
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
				{getDisplayText(value)}
			</button>
		{/if}
	{:else}
		<Popover bind:open>
			{#snippet children({ trigger, content })}
				<Button
					{...trigger}
					variant="outline"
					class={cn(
						"w-full justify-between",
						buttonClass
					)}
					{disabled}
				>
					<span class="truncate">
						{getDisplayText(value)}
					</span>
					<svg class="h-4 w-4 opacity-50" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
						<path d="m6 9 6 6 6-6"/>
					</svg>
				</Button>
				
				{#if open}
					<div
						{...content}
						class={cn(
							"w-full min-w-[200px] max-h-60 overflow-hidden",
							dropdownPosition === 'top' ? 'mb-1 bottom-full' : 'mt-1 top-full',
							content.class,
							contentClass
						)}
						onkeydown={handleKeydown}
					>
						{#if searchable}
							<div class="p-2 border-b">
								<Input
									bind:value={searchQuery}
									placeholder={searchPlaceholder}
									class="h-8 text-sm"
								/>
							</div>
						{/if}
						
						<div class="max-h-60 overflow-y-auto">
							{#if clearable && value}
								<button
									type="button"
									class={cn(
										"w-full px-3 py-2 text-left text-sm hover:bg-gray-100 focus:bg-gray-100 focus:outline-none border-b",
										optionClass
									)}
									onclick={clearSelection}
								>
									<span class="text-gray-500 italic">Clear selection</span>
								</button>
							{/if}
							
							{#each filteredOptions as option}
								{@const optValue = typeof option === 'string' ? option : option.value}
								{@const optLabel = typeof option === 'string' ? option : option.label || option.name || String(option.value)}
								<button
									type="button"
									class={cn(
										"w-full px-3 py-2 text-left text-sm hover:bg-gray-100 focus:bg-gray-100 focus:outline-none",
										value === optValue ? "bg-blue-50 text-blue-700" : "",
										optionClass
									)}
									onclick={() => selectOption(option)}
								>
									{optLabel}
								</button>
							{:else}
								<div class="px-3 py-2 text-sm text-gray-500 italic">
									No options found
								</div>
							{/each}
						</div>
					</div>
				{/if}
			{/snippet}
		</Popover>
	{/if}
</div>