<script>
  import { createEventDispatcher } from 'svelte';
  import emblaCarouselSvelte from 'embla-carousel-svelte';
  
  export let options = {};
  export let plugins = [];
  
  const dispatch = createEventDispatcher();
  
  let emblaApi;
  let canScrollPrev = false;
  let canScrollNext = false;
  
  function onInit(event) {
    emblaApi = event.detail;
    emblaApi.on('select', updateButtons);
    emblaApi.on('reInit', updateButtons);
    updateButtons();
    dispatch('init', emblaApi);
  }
  
  function updateButtons() {
    canScrollPrev = emblaApi.canScrollPrev();
    canScrollNext = emblaApi.canScrollNext();
  }
  
  export function scrollPrev() {
    if (emblaApi) emblaApi.scrollPrev();
  }
  
  export function scrollNext() {
    if (emblaApi) emblaApi.scrollNext();
  }
  
  export function scrollTo(index) {
    if (emblaApi) emblaApi.scrollTo(index);
  }
  
  export function getApi() {
    return emblaApi;
  }
</script>

<div class="relative">
  <div 
    class="overflow-hidden" 
    use:emblaCarouselSvelte={{ options, plugins }} 
    on:emblaInit={onInit}
  >
    <div class="flex">
      <slot />
    </div>
  </div>
  
  <slot name="controls" {canScrollPrev} {canScrollNext} {scrollPrev} {scrollNext} />
</div>