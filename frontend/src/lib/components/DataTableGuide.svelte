<script>
  import { Card } from "$lib/components/ui/card/index.js";
  import BasicDataTableExample from "./BasicDataTableExample.svelte";
</script>

<div class="space-y-6">
  <div>
    <h2 class="text-2xl font-bold mb-4">Advanced Data Table Component</h2>
    <p class="text-muted-foreground">
      A comprehensive data table built with TanStack Table and shadcn-svelte components.
    </p>
  </div>

  <div class="grid gap-6 md:grid-cols-2">
    <Card.Root>
      <Card.Header>
        <Card.Title>Features</Card.Title>
      </Card.Header>
      <Card.Content>
        <ul class="space-y-2 text-sm">
          <li>✅ Sorting (single and multi-column)</li>
          <li>✅ Filtering (global search and faceted filters)</li>
          <li>✅ Pagination with customizable page sizes</li>
          <li>✅ Row selection (single and multi-select)</li>
          <li>✅ Column visibility toggle</li>
          <li>✅ Row actions (view, edit, delete)</li>
          <li>✅ Responsive design</li>
          <li>✅ Customizable toolbar</li>
          <li>✅ TypeScript support</li>
          <li>✅ Accessible components</li>
        </ul>
      </Card.Content>
    </Card.Root>

    <Card.Root>
      <Card.Header>
        <Card.Title>Components Included</Card.Title>
      </Card.Header>
      <Card.Content>
        <ul class="space-y-2 text-sm">
          <li><code class="text-xs bg-muted px-1 py-0.5 rounded">DataTable</code> - Main table component</li>
          <li><code class="text-xs bg-muted px-1 py-0.5 rounded">DataTableActions</code> - Row action menu</li>
          <li><code class="text-xs bg-muted px-1 py-0.5 rounded">DataTableCheckbox</code> - Selection checkbox</li>
          <li><code class="text-xs bg-muted px-1 py-0.5 rounded">DataTableSortableHeader</code> - Sortable column header</li>
          <li><code class="text-xs bg-muted px-1 py-0.5 rounded">DataTablePagination</code> - Pagination controls</li>
          <li><code class="text-xs bg-muted px-1 py-0.5 rounded">DataTableToolbar</code> - Search and filters</li>
          <li><code class="text-xs bg-muted px-1 py-0.5 rounded">DataTableViewOptions</code> - Column visibility</li>
          <li><code class="text-xs bg-muted px-1 py-0.5 rounded">DataTableFacetedFilter</code> - Multi-select filters</li>
        </ul>
      </Card.Content>
    </Card.Root>
  </div>

  <Card.Root>
    <Card.Header>
      <Card.Title>Basic Usage</Card.Title>
    </Card.Header>
    <Card.Content>
      <pre class="text-xs bg-muted p-4 rounded-md overflow-x-auto"><code>{`<script>
  import { DataTable } from "$lib/components/data-table";
  import { columns } from "./columns.js";
  
  let data = [
    { id: "1", name: "John", email: "john@example.com" },
    { id: "2", name: "Jane", email: "jane@example.com" }
  ];
</script>

<DataTable 
  {data} 
  {columns}
  searchColumn="name"
  searchPlaceholder="Search names..."
  pageSize={10}
  onRowClick={(row) => console.log(row)}
  onRowSelectionChange={(rows) => console.log(rows)}
/>`}</code></pre>
    </Card.Content>
  </Card.Root>

  <Card.Root>
    <Card.Header>
      <Card.Title>Column Definition Example</Card.Title>
    </Card.Header>
    <Card.Content>
      <pre class="text-xs bg-muted p-4 rounded-md overflow-x-auto"><code>{`import { renderComponent, renderSnippet } from "$lib/components/ui/data-table";
import { DataTableSortableHeader, DataTableActions } from "$lib/components/data-table";

export const columns = [
  {
    id: "select",
    header: ({ table }) => renderComponent(DataTableCheckbox, {
      checked: table.getIsAllPageRowsSelected(),
      onCheckedChange: (value) => table.toggleAllPageRowsSelected(!!value)
    }),
    cell: ({ row }) => renderComponent(DataTableCheckbox, {
      checked: row.getIsSelected(),
      onCheckedChange: (value) => row.toggleSelected(!!value)
    })
  },
  {
    accessorKey: "name",
    header: ({ column }) => renderComponent(DataTableSortableHeader, {
      onclick: column.getToggleSortingHandler(),
      sortDirection: column.getIsSorted(),
      children: "Name"
    })
  },
  {
    id: "actions",
    cell: ({ row }) => renderComponent(DataTableActions, {
      id: row.original.id,
      onEdit: (id) => console.log("Edit", id),
      onDelete: (id) => console.log("Delete", id)
    })
  }
];`}</code></pre>
    </Card.Content>
  </Card.Root>

  <Card.Root>
    <Card.Header>
      <Card.Title>Advanced Features</Card.Title>
    </Card.Header>
    <Card.Content>
      <div class="space-y-4 text-sm">
        <div>
          <h4 class="font-medium mb-2">Faceted Filters</h4>
          <p class="text-muted-foreground mb-2">Add multi-select filters for specific columns:</p>
          <pre class="text-xs bg-muted p-2 rounded"><code>{`const facetedFilters = [
  {
    column: "status",
    title: "Status", 
    options: [
      { value: "active", label: "Active", icon: CheckIcon },
      { value: "inactive", label: "Inactive", icon: XIcon }
    ]
  }
];`}</code></pre>
        </div>
        
        <div>
          <h4 class="font-medium mb-2">Custom Cell Rendering</h4>
          <p class="text-muted-foreground mb-2">Use createRawSnippet for simple HTML rendering:</p>
          <pre class="text-xs bg-muted p-2 rounded"><code>{`cell: ({ row }) => {
  const snippet = createRawSnippet<[string]>((getValue) => {
    const value = getValue();
    return {
      render: () => \`<div class="font-bold">\${value}</div>\`
    };
  });
  return renderSnippet(snippet, row.getValue("fieldName"));
}`}</code></pre>
        </div>
      </div>
    </Card.Content>
  </Card.Root>

  <!-- Basic Example -->
  <div class="border-t pt-6">
    <BasicDataTableExample />
  </div>
</div>