<!--
  @fileoverview Email Settings component for email search and integration configuration
-->
<script>

  import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
  } from "../ui/card/index.js";
  import { Label } from "../ui/label/index.js";
  import { Input } from "../ui/input/index.js";
  import { Checkbox } from "../ui/checkbox/index.js";

  export let settings = {};
  export let onchange;

  // Email setting keys from backend ConfigKeys
  const EMAIL_KEYS = {
    SEARCH_DAYS_BUFFER: "email.search_days_buffer",
    CUSTOM_SEARCH_STRING: "email.custom_search_string",
    INCLUDE_PAYEE_BY_DEFAULT: "email.include_payee_by_default",
    INCLUDE_AMOUNT_BY_DEFAULT: "email.include_amount_by_default",
    APPEND_URL_TO_MEMO: "email.append_url_to_memo",
    BULK_AUTO_ATTACH_SINGLE_RESULT: "email.bulk_auto_attach_single_result",
  };

  // Get setting value helper
  function getSettingValue(key, defaultValue = "") {
    const setting = settings[key];
    if (!setting || !setting.value) return defaultValue;

    // Extract value from ConfigValue union
    const value = setting.value;
    return (
      value.stringValue ??
      value.intValue ??
      value.doubleValue ??
      value.boolValue ??
      defaultValue
    );
  }

  // Update setting helper
  function updateSetting(key, value, description = "") {
    const currentValue = getSettingValue(key);
    
    // Only update if the value has actually changed
    if (currentValue !== value) {
      onchange?.({ detail: { key, value, type: "Email", description } });
    }
  }
</script>

<div
  class="email-settings columns-1 lg:columns-2 gap-6 space-y-6"
  data-testid="email-settings"
>
  <!-- Email Search Configuration -->
  <Card class="break-inside-avoid mb-6">
    <CardHeader>
      <CardTitle>Email Search Configuration</CardTitle>
      <CardDescription
        >Configure how emails are searched and matched with transactions</CardDescription
      >
    </CardHeader>
    <CardContent class="space-y-4">
      <div class="space-y-2">
        <Label for="email-search-days">Search Days Buffer</Label>
        <Input
          id="email-search-days"
          type="number"
          min="1"
          max="30"
          value={getSettingValue(EMAIL_KEYS.SEARCH_DAYS_BUFFER, 3)}
          onblur={(e) =>
            updateSetting(
              EMAIL_KEYS.SEARCH_DAYS_BUFFER,
              parseInt(e.target.value),
              "Number of days before and after transaction date to search for emails",
            )}
          onkeydown={(e) => {
            if (e.key === "Enter") {
              updateSetting(
                EMAIL_KEYS.SEARCH_DAYS_BUFFER,
                parseInt(e.target.value),
                "Number of days before and after transaction date to search for emails",
              );
            }
          }}
          placeholder="Days to search before/after transaction date"
          data-testid="email-search-days-input"
        />
        <p class="text-sm text-muted-foreground">
          Number of days before and after transaction date to search for emails
        </p>
      </div>

      <div class="space-y-2">
        <Label for="custom-search-string"
          >Custom Search String With Templates</Label
        >
        <Input
          id="custom-search-string"
          type="text"
          value={getSettingValue(EMAIL_KEYS.CUSTOM_SEARCH_STRING, "")}
          onblur={(e) =>
            updateSetting(
              EMAIL_KEYS.CUSTOM_SEARCH_STRING,
              e.target.value,
              "Custom search string with variables: &#123;payee&#125;, &#123;amount&#125;, &#123;date&#125;",
            )}
          onkeydown={(e) => {
            if (e.key === "Enter") {
              updateSetting(
                EMAIL_KEYS.CUSTOM_SEARCH_STRING,
                e.target.value,
                "Custom search string with variables: &#123;payee&#125;, &#123;amount&#125;, &#123;date&#125;",
              );
            }
          }}
          placeholder="e.g., &#123;payee&#125; receipt &#123;amount&#125;"
          data-testid="custom-search-string-input"
        />
        <p class="text-sm text-muted-foreground">
          Available variables: <code>&#123;payee&#125;</code>,
          <code>&#123;amount&#125;</code>, <code>&#123;date&#125;</code>. Leave
          empty to use default search logic.
        </p>
      </div>
    </CardContent>
  </Card>

  <!-- Search Behavior Settings -->
  <Card class="break-inside-avoid mb-6">
    <CardHeader>
      <CardTitle>Search Behavior</CardTitle>
      <CardDescription
        >Configure default search behavior and content inclusion</CardDescription
      >
    </CardHeader>
    <CardContent class="space-y-4">
      <div class="flex items-center space-x-2">
        <Checkbox
          id="include-payee"
          checked={getSettingValue(EMAIL_KEYS.INCLUDE_PAYEE_BY_DEFAULT, true)}
          onCheckedChange={(checked) =>
            updateSetting(
              EMAIL_KEYS.INCLUDE_PAYEE_BY_DEFAULT,
              checked,
              "Include payee in email search by default",
            )}
          data-testid="include-payee-checkbox"
        />
        <Label
          for="include-payee"
          class="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
        >
          Include payee in email search to improve matching accuracy
        </Label>
      </div>

      <div class="flex items-center space-x-2">
        <Checkbox
          id="include-amount"
          checked={getSettingValue(EMAIL_KEYS.INCLUDE_AMOUNT_BY_DEFAULT, true)}
          onCheckedChange={(checked) =>
            updateSetting(
              EMAIL_KEYS.INCLUDE_AMOUNT_BY_DEFAULT,
              checked,
              "Include amount in email search by default",
            )}
          data-testid="include-amount-checkbox"
        />
        <Label
          for="include-amount"
          class="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
        >
          Include transaction amount in email search to improve matching
          accuracy
        </Label>
      </div>

      <div class="flex items-center space-x-2">
        <Checkbox
          id="append-url"
          checked={getSettingValue(EMAIL_KEYS.APPEND_URL_TO_MEMO, true)}
          onCheckedChange={(checked) =>
            updateSetting(
              EMAIL_KEYS.APPEND_URL_TO_MEMO,
              checked,
              "Append email URL to memo when only 1 email is found",
            )}
          data-testid="append-url-checkbox"
        />
        <Label
          for="append-url"
          class="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
        >
          Append email URL to memo when exactly one email is found
        </Label>
      </div>

      <div class="flex items-center space-x-2">
        <Checkbox
          id="bulk-auto-attach"
          checked={getSettingValue(EMAIL_KEYS.BULK_AUTO_ATTACH_SINGLE_RESULT, true)}
          onCheckedChange={(checked) =>
            updateSetting(
              EMAIL_KEYS.BULK_AUTO_ATTACH_SINGLE_RESULT,
              checked,
              "When bulk searching, auto-attach email when there's only 1 result",
            )}
          data-testid="bulk-auto-attach-checkbox"
        />
        <Label
          for="bulk-auto-attach"
          class="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
        >
          When bulk searching, auto-attach email when there's only 1 result
        </Label>
      </div>
    </CardContent>
  </Card>

  <!-- Email Search Tips -->
  <Card class="break-inside-avoid mb-6">
    <CardHeader>
      <CardTitle>Search Tips</CardTitle>
      <CardDescription
        >Best practices for email search configuration</CardDescription
      >
    </CardHeader>
    <CardContent>
      <div class="space-y-3 text-sm">
        <div>
          <strong>Search Days Buffer:</strong> A larger buffer (5-7 days) helps catch
          emails that arrive before or after the transaction date, but may return
          more irrelevant results.
        </div>
        <div>
          <strong>Custom Search String:</strong> Use templates with available
          variables: <code>&#123;payee&#125;</code> (merchant name),
          <code>&#123;amount&#125;</code>
          (e.g., $4.50), <code>&#123;date&#125;</code> (YYYY-MM-DD). Example:
          <code>"&#123;payee&#125; receipt &#123;amount&#125;"</code>.
        </div>
        <div>
          <strong>Payee Inclusion:</strong> Including payee names helps filter results
          but may miss emails where the merchant name differs from the YNAB payee
          name.
        </div>
        <div>
          <strong>Amount Inclusion:</strong> Including amounts is very effective
          for matching but may miss emails that don't contain the exact amount (e.g.,
          partial charges).
        </div>
      </div>
    </CardContent>
  </Card>
</div>
