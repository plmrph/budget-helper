<script>
	import {
		Check as CheckIcon,
		ChevronsUpDown as ChevronsUpDownIcon,
	} from "@lucide/svelte";
	import * as Command from "$lib/components/ui/command/index.js";
	import * as Popover from "$lib/components/ui/popover/index.js";
	import { cn } from "$lib/utils.js";

	let {
		value = $bindable(""),
		open = $bindable(false),
		options = [],
		placeholder = "Select an option...",
		searchPlaceholder = "Search...",
		emptyMessage = "No results found.",
		displayValue = null,
		disabled = false,
		class: className = "",
		onSelect = null,
		inline = false,
		...restProps
	} = $props();

	// Get display text for selected value
	function getDisplayText(val) {
		// Use provided displayValue if available (prevents flickering during updates)
		if (displayValue !== null && displayValue !== undefined) {
			return displayValue || placeholder;
		}
		
		if (!val) return placeholder;

		// Find the option to get its label
		const option = options.find((opt) => {
			const optValue = typeof opt === "string" ? opt : opt.value;
			return optValue === val;
		});

		if (option) {
			return typeof option === "string"
				? option
				: option.label || option.name || String(option.value);
		}

		return String(val);
	}

	// Handle option selection – keep UUID for backend
	function selectOption(option) {
		const optValue = typeof option === "string" ? option : option.value;
		value = optValue;
		open = false;

		if (onSelect) {
			onSelect(optValue, option);
		}
	}
</script>

<div class={cn("relative", className)} {...restProps}>
	<Popover.Root bind:open>
		<Popover.Trigger>
			<!-- Button styled to look like plain text -->
			<button
				type="button"
				class="w-full text-sm truncate text-muted-foreground cursor-pointer hover:bg-gray-50 rounded px-1 py-1 flex items-center justify-between border-0 bg-transparent focus:outline-none"
				{disabled}
				aria-label={`Edit: ${getDisplayText(value)}`}
			>
				<span class="truncate">{getDisplayText(value)}</span>
				<ChevronsUpDownIcon class="ml-2 h-4 w-4 shrink-0 opacity-50" />
			</button>
		</Popover.Trigger>
		<Popover.Content class="w-[200px] p-0" align="start">
			<Command.Root>
				<Command.Input
					placeholder={searchPlaceholder}
					class="h-9 pl-3"
				/>
				<Command.List>
					<Command.Empty>{emptyMessage}</Command.Empty>
					<Command.Group>
						{#each options as option (typeof option === "string" ? option : option.value)}
							{@const optValue = typeof option === "string" ? option : option.value}
							{@const optLabel = typeof option === "string" ? option : option.label || option.name || String(option.value)}
							{@const commandValue = `${optLabel} — ${optValue}`}
							<Command.Item
								value={commandValue}
								onSelect={() => selectOption(option)}
							>
								<CheckIcon
									class={cn(
										"mr-2 h-4 w-4",
										value === optValue ? "opacity-100" : "opacity-0"
									)}
								/>
								{optLabel}
							</Command.Item>
						{/each}
					</Command.Group>
				</Command.List>
			</Command.Root>
		</Popover.Content>
	</Popover.Root>
</div>
