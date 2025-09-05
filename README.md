# Budget Helper

Budget Helper is a fully local budget application designed to enhance your YNAB (You Need A Budget) workflow. It takes a privacy-focused approach where all your data is stored and used locally, so the only outside connections are: pulling transactions from the YNAB API, pushing transaction updates back to the YNAB API, and pulling email information (if email search is set up). It provides a Sync Budget view which prevents any unexpected data changes, single/batch email searching with auto-attachment of emails, and single/batch predicting of Categories using your own trained ML model.

Key features include:
1. **Fully Local** - All your financial data stays on your machine (other than when you choose to send transaction updates to YNAB.)
2. **YNAB Integration** - Seamless editing and synchronization with your YNAB budget, uses batch APIs to not exceed rate limits  
3. **Manual Sync Control** - Sync Budget view with verification so your YNAB records aren't updated unexpectedly
4. **Email Searching** - Automatically find and attach relevant emails to transactions when criteria are met
5. **ML Predictions** - Advanced machine learning for intelligent transaction categorization, learns various intricate patterns between your data. See the ML readme for more info ([ML README](backend/business_layer/README.md))

## Quick Start

Install Docker if you haven't already https://docs.docker.com/get-started/get-docker/

#### Use pre-built images
```bash
git clone https://github.com/plmrph/budget-helper.git
cd budget-helper
cp .env.frontend.example .env.frontend
cp .env.backend.example .env.backend
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up
```

The application will be available at http://localhost

#### (optional alternative) Building images from source
```bash
git clone https://github.com/plmrph/budget-helper.git
cd budget-helper
cp .env.frontend.example .env.frontend
cp .env.backend.example .env.backend
docker compose build
docker compose up
```

### YNAB Setup

YNAB integration is necessary for core transaction functionality. To set this up:

1. Get your YNAB Personal Access Token:
   - Go to https://app.youneedabudget.com/settings/developer
   - Generate a new Personal Access Token
2. In the Budget Helper UI, go to Settings → YNAB Configuration
3. Enter your Personal Access Token and test the connection
4. Select your budget from the dropdown
5. Go to the Transactions page and press "Sync Budget"
  1. Select the date for how far back to sync (if you're planning to use ML, I recommend 24 months)
  2. Press "Refresh Data"
  3. Review what data is going to be synced in. Leave "<" for the ones you want to import and "X" for any you don't
  4. Press "Confirm" to sync
6. Now you can edit Transactions in the grid and sync them back to YNAB whenever you'd like

### Gmail Setup (Optional)

Gmail integration enables searching for relevant emails related to your transactions, including batch search functionality for all transactions on the current page.

To get OAuth credentials:
1. Go to Google Cloud Console [https://console.cloud.google.com/projectcreate](https://console.cloud.google.com/projectcreate)
2. Create a project
3. Enable Gmail API
4. Create OAuth 2.0 credentials
5. Add [http://localhost](http://localhost/) to authorized origins
6. Add [http://localhost/api/auth/gmail/callback](http://localhost/api/auth/gmail/callback) to redirect URIs
7. Go to [https://console.cloud.google.com/auth/audience](https://console.cloud.google.com/auth/audience) and add your email under "Test users"
8. In the Budget Helper UI, go to Settings → Gmail Configuration
9. Enter your Client ID and Client Secret, then connect your account
10. You'll get a message like "You’ve been given access to an app that’s currently being tested. You should only continue if you know the developer that invited you.". Press continue, and continue
> If you get a message like "has not completed the Google verification process. The app is currently being tested, and can only be accessed by developer-approved testers. If you think you should have access, contact the developer." then you haven't done step 7

### ML Setup (Optional)

Machine learning enables automatic category prediction for transactions. The workflow involves creating a dataset from your existing data, training the model (typically 2-15 minutes), then using it for predictions including batch predictions.

To set up ML:
1. Go to the ML Training page in the application (button in the top right)
2. Create a dataset from your transactions (recommended: at least 500-1000 categorized transactions for good performance. If you didn't import enough Transactions when Syncing the Budget, go back and import more)
3. Train a model using the PXBlendSC strategy
4. Once training completes, review the performance metrics of the model. If they look good, set it as your default model on the Model Management page
5. Use predictions on the Transactions page for individual or batch categorization

For detailed ML configuration and strategy information, see the [ML README](backend/business_layer/README.md).


## Test Data
A helper script is available to populate or wipe demo data in Postgres if you just wanted to try out the UI before connecting anything.

- Seed: creates 1 demo budget, some categories/payees/accounts, and 200 transactions. Of these, 10 are unapproved with a category and 10 are unapproved with no category.
- Wipe: deletes user data (budgets, accounts, categories, payees, transactions, metadata, history, file_entities) but preserves system tables like `config_items` and `schema_migrations`.

Run the backend container first, and then run the script through it using these commands (uses DATABASE_URL from .env.backend):

```bash
docker compose exec backend python -m scripts.test_data seed
docker compose exec backend python -m scripts.test_data wipe
```

## Key Features In Detail

### Fully Local Data Control
Machine Learning models are trained and stored locally, and all data you see on screen is stored in a local Postgres database. The only time information leaves the app is when you choose to send transaction updates to YNAB.
Your financial data never leaves your machine. The application runs entirely in Docker containers on your local system, ensuring complete privacy and control over your sensitive information.

![Transaction Grid](screenshots/Transaction%20Grid.png)

### YNAB Integration with Manual Sync
- **Sync Budget**: Click the "Sync Budget" button to bring up the diff view, which shows you exactly what changes will be made. You can review incoming changes from YNAB before applying them locally, or preview what changes will be sent to YNAB before committing them.
- **Verification workflow**: All synchronization requires explicit user approval to prevent accidental data overwrites
- **Selective sync**: Choose which specific transactions to import or export

![Sync Budget](screenshots/Sync%20Budget.jpg)

### Email Search and Auto-Attachment
- **Individual search**: Find emails related to specific transactions using payee, amount, and date information by clicking on the email icon in a Transaction's row
- **Batch processing**: Search emails for all transactions currently displayed on the page by pressing the Email Search button
- **Auto-attachment**: Automatically attach email URLs to transaction memos when single results are found
- **Flexible search**: Customize search parameters including date ranges and search terms

![Email Popup](screenshots/Email%20popup.jpg)

### Machine Learning Predictions
- **PXBlendSC Strategy**: Advanced ensemble model combining LightGBM and SVM with sophisticated feature engineering
- **Pattern Recognition**: Learns from payee names, amounts, dates, account patterns, and transaction combinations
- **Adaptive Thresholds**: Smart abstention when confidence is low to avoid incorrect categorizations during training. However during prediction the model will always give a result and the confidence level floor is 10%.
- **Batch Predictions**: Categorize multiple transactions at once with confidence scores
- **Continuous Learning**: Retrain models as your transaction history grows
- **Switch between models**: experiment and swap between different models easily.

![Prepare Dataset](screenshots/Prepare%20Dataset.png)
![Model Setup](screenshots/Train%20Model%20Setup.png)
![Model Training In Progess](screenshots/Training%20In-Progress.png)
![Model Training Done](screenshots/Training%20Done.png)
![Model Metrics](screenshots/Model%20Performance.png)

### More Screenshots
[Screenshots Folder](screenshots/)

## Issues?
If you're getting some sorts of errors
1. Check your Browser's Console logs (CMD+Option+I for Chrome on macOS)
2. Check for errors on the backend by running `docker compose logs backend | grep "ERROR"`
3. Check GitHub Issues if someone already reported the problem and has a solution
4. If not, please open an Issue and
  1. Describe the steps you took to reproduce/encounter the issue
  2. Remove any sensitive information from your console + backend logs
  3. Add the logs to the issue

## Development

For convenience of use and set up, the whole app is containerized.

### Prerequisites

- Docker
- Git

### Quick Commands
```bash
# Start the application
docker compose up

# Stop the application  
docker compose down

# Stop and remove all data
# WARNING: THIS WILL DELETE ALL YOUR LOCAL DATA PERMANENTLY. Only do this if you want a fresh start.
docker compose down -v
```

### Project Structure
```
budget-helper/
├── frontend/          # Svelte application with modern UI components  
├── backend/           # FastAPI application with clean architecture
├── .llm/              # System design and thrift interfaces between components. Also contains the LLM guidance that was used to build this app for those who are curious
├── docker-compose.yml # Multi-container orchestration
└── nginx.conf         # Reverse proxy configuration
```

### Development Features
- **Auto-reload**: Backend automatically restarts when code changes are detected (warning, container restarts will interrupt any in-progress ML Training)
- **Hot module replacement**: Frontend updates instantly during development
- **Container-first**: All development happens within Docker containers for consistency
- **Database migrations**: Automatically applied on startup

### Dependency Management

Python dependencies are baked into the backend image and installed into `/opt/venv` during the image build. This avoids re-downloading packages every time you `docker compose up`.

When you change dependencies (edit `backend/pyproject.toml` or `backend/uv.lock`):
- `docker compose build backend` - Rebuild the backend image
- `docker compose build --no-cache backend` - Force a clean rebuild if lockfile changed

### Extensibility
The application is designed for extensibility through well-defined interfaces:

- **Budget Platform**: Implement `BudgetingPlatformAccess` interface to support budget platforms beyond YNAB
- **Email Platform**: Implement `MetadataSourceAccess` interface to add support for email providers beyond Gmail  
- **ML Models**: Implement `MLModelStrategy` interface to add new machine learning approaches beyond PXBlendSC

For detailed implementation guidance, see the [Backend README](backend/README.md) and [Frontend README](frontend/README.md).