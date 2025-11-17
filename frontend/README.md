# Budget Helper Frontend

The frontend is a modern Svelte 5 application with shadcn-svelte components, providing an interface for budget management, transaction categorization, and ML-powered predictions.

## Technology Stack

- **Framework**: Svelte 5 with modern reactive syntax
- **Build Tool**: Vite 7 for fast development and optimized production builds
- **UI Components**: shadcn-svelte component library
- **Styling**: Tailwind CSS v3 with custom design system
- **Routing**: svelte-spa-router for client-side navigation
- **Package Manager**: npm
- **Type Safety**: JSDoc annotations throughout

## Project Structure

```
frontend/
├── src/
│   ├── lib/
│   │   ├── components/
│   │   │   ├── ui/                  # shadcn-svelte components
│   │   │   ├── Navigation.svelte
│   │   │   └── editable-cells/      # Transaction grid editing components
│   │   ├── pages/                   # Main application pages
│   │   │   ├── Dashboard.svelte
│   │   │   ├── Transactions.svelte
│   │   │   ├── Settings.svelte
│   │   │   └── MLTraining.svelte
│   │   ├── stores/                  # Svelte stores for state management
│   │   │   ├── bulkMLPredict.js
│   │   │   └── transactionSync.js
│   │   ├── api/
│   │   │   └── client.js            # Backend API integration
│   │   └── utils.js                 # Utility functions
│   ├── App.svelte                   # Root component with routing
│   ├── main.js                      # Application entry point
│   └── app.css                      # Global styles and Tailwind imports
├── components.json                  # shadcn-svelte configuration
├── tailwind.config.js               # Tailwind CSS configuration
├── vite.config.js                   # Vite build configuration
├── jsconfig.json                    # JavaScript/IDE configuration
└── package.json                     # Dependencies and build scripts
```

## Pages and Features

### Transactions Page
The core of the application featuring a data grid with powerful interaction capabilities:

#### Transaction Grid Features
- **Sortable Columns**: Click headers to sort by any field (date, amount, payee, category)
- **Inline Editing**: Edit transactions directly in the grid
- **Search and Filtering**: Real-time search across all transaction fields
- **Pagination**: Efficient handling of large transaction datasets

#### Sync Budget Functionality
- **Sync Budget**: Click the "Sync Budget" button to bring up the diff view, which shows you exactly what changes will be made
- **Verification workflow**: All synchronization requires explicit user approval to prevent accidental data overwrites
- **Selective sync**: Choose specific transactions to import or export

#### ML Integration
- **Individual Predictions**: Get category suggestions for single transactions, click the prediction to see the top 3 predictions
- **Batch Predictions**: Predict categories for all visible transactions
- **Confidence Indicators**: Visual confidence scores for each prediction
- **One-Click Apply**: Accept the predicted category and approve the transaction with a single click

### Settings Page
Centralized configuration for all application integrations:

#### YNAB Configuration
- **API Key Management**: Secure storage and testing of YNAB Personal Access Token
- **Budget Selection**: Choose active budget from available YNAB budgets
- **Sync Preferences**: Configure automatic vs manual sync behavior
- **Connection Testing**: Verify YNAB connectivity and permissions

#### Gmail Configuration  
- **OAuth Setup**: Secure Google OAuth2 integration
- **Search Preferences**: Configure email search parameters and date ranges
- **Auto-Attachment Settings**: Control when emails are automatically attached to transactions
- **Batch Search Options**: Configure bulk email search behavior

#### AI/ML Settings
- **Training Parameters**: Configure dataset size and training duration

### ML Training Page
Comprehensive interface for machine learning model management:

#### Dataset Creation
- **Transaction Selection**: Choose date ranges and transaction types for training data
- **Category Balance**: Visualize category distribution in training dataset
- **Quality Metrics**: Dataset size recommendations and quality indicators

#### Model Training
- **Strategy Selection**: Choose ML approach (currently PXBlendSC)
- **Progress Monitoring**: Real-time training progress with time estimates
- **Performance Metrics**: Cross-validation scores and accuracy measurements
- **Training History**: Track and compare multiple training runs

#### Model Management
- **Model Library**: Browse and manage trained models
- **Performance Comparison**: Compare accuracy across different models
- **Deployment**: Set models as default for predictions

## Stores and State Management

### Bulk ML Predict Store (`bulkMLPredict.js`)
Manages batch ML prediction workflows:
- **Transaction Selection**: Track which transactions are selected for batch prediction
- **Prediction Results**: Store and manage prediction results with confidence scores
- **Application State**: Handle bulk application of predictions
- **Error Handling**: Manage prediction failures and retry logic

### Transaction Sync Store (`transactionSync.js`)
Coordinates YNAB synchronization:
- **Diff Management**: Track differences between local and remote data
- **Sync State**: Manage import/export workflow state
- **Conflict Resolution**: Handle synchronization conflicts
- **Progress Tracking**: Monitor sync operation progress

## Component Architecture

### UI Components (`lib/components/ui/`)
Built on shadcn-svelte for consistency and accessibility:
- **Forms**: Consistent form controls with validation
- **Tables**: Data grid components with sorting and pagination
- **Dialogs**: Modal dialogs for confirmations and detailed views
- **Buttons**: Standardized button variants and states
- **Input Controls**: Text inputs, selects, checkboxes with validation

### Editable Cells (`lib/components/ui/editable-cells/`)
Specialized components for in-grid editing:
- **EditableMLCategoryCell**: ML-powered category editing with predictions
- **EditableTextCell**: Inline text editing with validation
- **EditableSelectCell**: Dropdown selection within grid cells
- **EditableAmountCell**: Formatted currency input with validation

## API Integration

The frontend includes a comprehensive API client (`src/lib/api/client.js`) providing:

### Transaction Management
- **CRUD Operations**: Create, read, update, delete transactions
- **Bulk Operations**: Batch updates and predictions
- **Sync Operations**: Import/export with YNAB
- **Search and Filtering**: Server-side transaction queries

### Email Integration
- **Search API**: Find emails related to transactions
- **Batch Search**: Search emails for multiple transactions
- **Attachment Management**: Link emails to transaction records
- **OAuth Management**: Handle Gmail authentication flow

### AI/ML Operations
- **Model Training**: Train new ML models with custom datasets
- **Prediction API**: Get category predictions for transactions
- **Model Management**: List, deploy, and manage trained models
- **Performance Metrics**: Retrieve model accuracy and performance data

### Settings Management
- **Configuration CRUD**: Manage all application settings
- **Connection Testing**: Verify external service connections
- **Authentication**: Manage OAuth tokens and API keys

## Development Workflow

### Available Scripts
```bash
pnpm run dev      # Start development server with hot reloading
pnpm run build    # Build production version
pnpm run preview  # Preview production build locally
pnpm run type-check # Run JSDoc type checking
```

### Development Features
- **Hot Module Replacement**: Instant updates during development
- **API Proxy**: Development server proxies `/api` requests to backend
- **Error Overlay**: Clear error reporting during development
- **Path Aliases**: Clean imports using `$lib` alias

### Code Organization
- **Component-First**: UI broken into reusable components
- **Store-Based State**: Reactive stores for complex state management
- **API Abstraction**: Centralized API client prevents direct fetch usage
- **Type Annotations**: JSDoc provides IntelliSense and error checking

## Styling and Design

### Tailwind CSS Integration
- **Design System**: Consistent spacing, colors, and typography
- **Responsive Design**: Mobile-first responsive layouts
- **Dark Mode Support**: Automatic dark/light theme switching
- **Component Variants**: Standardized component styling patterns

### shadcn-svelte Components
- **Accessibility**: Built-in ARIA labels and keyboard navigation
- **Theming**: Consistent design tokens across all components
- **Customization**: Easy customization through CSS variables
- **Performance**: Optimized components with minimal bundle impact

## Performance Considerations

### Optimization Strategies
- **Virtual Scrolling**: Efficient rendering of large transaction lists
- **Lazy Loading**: Components loaded only when needed
- **Debounced Search**: Prevent excessive API calls during search
- **Caching**: Smart caching of API responses and computed values

### Bundle Optimization
- **Tree Shaking**: Eliminate unused code from production builds
- **Code Splitting**: Load routes and heavy components on demand
- **Asset Optimization**: Compressed images and optimized fonts
- **Modern Builds**: ES modules for modern browsers