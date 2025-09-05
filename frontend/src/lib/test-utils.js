/**
 * @fileoverview Test utilities for Svelte component testing
 * Provides helpers for testing UI components, forms, and interactions
 */

import { tick } from 'svelte';

/**
 * Wait for DOM updates and component re-renders
 * @param {number} ms - Milliseconds to wait
 * @returns {Promise<void>}
 */
export async function waitFor(ms = 0) {
  await tick();
  if (ms > 0) {
    await new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Simulate user input on an element
 * @param {HTMLElement} element - Input element
 * @param {string} value - Value to input
 * @returns {Promise<void>}
 */
export async function userInput(element, value) {
  element.value = value;
  element.dispatchEvent(new Event('input', { bubbles: true }));
  await tick();
}

/**
 * Simulate user click on an element
 * @param {HTMLElement} element - Element to click
 * @returns {Promise<void>}
 */
export async function userClick(element) {
  element.dispatchEvent(new Event('click', { bubbles: true }));
  await tick();
}

/**
 * Simulate form submission
 * @param {HTMLFormElement} form - Form element
 * @returns {Promise<void>}
 */
export async function submitForm(form) {
  form.dispatchEvent(new Event('submit', { bubbles: true }));
  await tick();
}

/**
 * Check if element is visible
 * @param {HTMLElement} element - Element to check
 * @returns {boolean}
 */
export function isVisible(element) {
  const style = window.getComputedStyle(element);
  return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
}

/**
 * Get element by test ID
 * @param {string} testId - Test ID attribute
 * @param {HTMLElement} [container] - Container to search within
 * @returns {HTMLElement|null}
 */
export function getByTestId(testId, container = document) {
  return container.querySelector(`[data-testid="${testId}"]`);
}

/**
 * Get all elements by test ID
 * @param {string} testId - Test ID attribute
 * @param {HTMLElement} [container] - Container to search within
 * @returns {NodeList}
 */
export function getAllByTestId(testId, container = document) {
  return container.querySelectorAll(`[data-testid="${testId}"]`);
}

/**
 * Check if element has specific class
 * @param {HTMLElement} element - Element to check
 * @param {string} className - Class name to check for
 * @returns {boolean}
 */
export function hasClass(element, className) {
  return element.classList.contains(className);
}

/**
 * Mock API response
 * @param {Object} response - Response object
 * @returns {Promise<Object>}
 */
export function mockApiResponse(response) {
  return Promise.resolve({
    success: true,
    data: response
  });
}

/**
 * Mock API error
 * @param {string} error - Error message
 * @returns {Promise<Object>}
 */
export function mockApiError(error) {
  return Promise.resolve({
    success: false,
    error
  });
}

/**
 * Test responsive breakpoints
 * @param {HTMLElement} element - Element to test
 * @param {Object} breakpoints - Breakpoint configurations
 * @returns {Object} Test results for each breakpoint
 */
export function testResponsiveBreakpoints(element, breakpoints = {
  mobile: 375,
  tablet: 768,
  desktop: 1024
}) {
  const results = {};
  
  Object.entries(breakpoints).forEach(([name, width]) => {
    // Simulate viewport width
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: width,
    });
    
    // Trigger resize event
    window.dispatchEvent(new Event('resize'));
    
    // Check element visibility and layout
    const computedStyle = window.getComputedStyle(element);
    results[name] = {
      width,
      visible: isVisible(element),
      display: computedStyle.display,
      flexDirection: computedStyle.flexDirection,
      gridTemplateColumns: computedStyle.gridTemplateColumns
    };
  });
  
  return results;
}