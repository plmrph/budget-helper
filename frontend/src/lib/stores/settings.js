/**
 * @fileoverview Settings store that fetches and caches application settings
 */

import { writable } from 'svelte/store';
import { settingsApi } from '../api/client.js';

function createSettingsStore() {
  const { subscribe, set, update } = writable({
    loaded: false,
    loading: false,
    data: null,
    error: null
  });

  async function loadAll(force = false) {
    let current;
    update(s => (current = s) || s);
    if (current && current.loaded && !force) return current.data;

    update(s => ({ ...s, loading: true, error: null }));

    try {
      const resp = await settingsApi.getAll();
      if (resp && resp.success && resp.data && resp.data.settings) {
        update(s => ({ ...s, data: resp.data.settings, loaded: true, loading: false }));
        return resp.data.settings;
      }

      const msg = (resp && (resp.message || resp.error)) || 'Failed to load settings';
      update(s => ({ ...s, error: msg, loading: false }));
      return null;
    } catch (err) {
      console.error('settings store loadAll error:', err);
      update(s => ({ ...s, error: err?.message || String(err), loading: false }));
      return null;
    }
  }

  function getSetting(key) {
    let value = null;
    update(s => {
      if (s && s.data) {
        value = s.data[key];
      }
      return s;
    });
    return value;
  }
  
  function extractValue(settingEntry, defaultValue = null) {
    if (!settingEntry) return defaultValue;
    const val = settingEntry.value || settingEntry;
    if (val == null) return defaultValue;

  if (val.stringValue != null) return val.stringValue;
  if (val.intValue != null) return val.intValue;
  if (val.doubleValue != null) return val.doubleValue;
  if (val.floatValue != null) return val.floatValue;
  if (val.boolValue != null) return val.boolValue;
  if (val.stringList != null) return val.stringList;
  if (val.stringMap != null) return val.stringMap;
  if (val.syncState != null) return val.syncState;

    return defaultValue;
  }

  async function getSettingValue(key) {
    let cached = null;
    update(s => {
      if (s && s.loaded && s.data && Object.prototype.hasOwnProperty.call(s.data, key)) {
        cached = s.data[key];
      }
      return s;
    });
    if (cached !== null && cached !== undefined) return extractValue(cached, null);

    const data = await loadAll();
    if (!data) return null;
    const entry = data[key];
    return extractValue(entry, null);
  }

  async function refresh() {
    return loadAll(true);
  }

  // Eagerly load settings once
  loadAll().catch(() => {});

  return {
    subscribe,
    loadAll,
    getSetting,
    getSettingValue,
    refresh
  };
}

export const settingsStore = createSettingsStore();
