<script>
	import { cn } from "$lib/utils.js";

	let {
		open = $bindable(false),
		children,
		class: className,
		...restProps
	} = $props();

	let triggerElement;
	let contentElement;

	function handleTriggerClick() {
		open = !open;
	}

	function handleClickOutside(event) {
		if (
			open &&
			contentElement &&
			!contentElement.contains(event.target) &&
			triggerElement &&
			!triggerElement.contains(event.target)
		) {
			open = false;
		}
	}
</script>

<svelte:window onclick={handleClickOutside} />

<div class="relative inline-block" {...restProps}>
	{@render children?.({
		trigger: {
			element: triggerElement,
			onclick: handleTriggerClick,
			"aria-expanded": open,
			"aria-haspopup": "true",
		},
		content: {
			element: contentElement,
			class: cn(
				"absolute z-[9999] min-w-[8rem] overflow-hidden rounded-md border bg-white p-1 shadow-lg",
				"data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95",
				className,
			),
			"data-state": open ? "open" : "closed",
		},
	})}
</div>
