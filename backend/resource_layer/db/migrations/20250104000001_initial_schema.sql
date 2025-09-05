-- migrate:up

-- Create enum types based on thrift definitions
CREATE TYPE entity_type AS ENUM (
    'transaction',
    'budget',
    'account',
    'category',
    'payee',
    'metadata',
    'config_item',
    'model_info',
    'file_entity'
);

CREATE TYPE budgeting_platform_type AS ENUM ('YNAB');
CREATE TYPE metadata_type AS ENUM ('Email', 'Prediction');
CREATE TYPE email_platform_type AS ENUM ('Gmail');
CREATE TYPE model_type AS ENUM ('PXBlendSC');
CREATE TYPE config_type AS ENUM ('System', 'Email', 'AI', 'Display', 'ExternalSystem');
CREATE TYPE sync_status AS ENUM ('Success', 'Pending', 'Partial', 'Fail');
CREATE TYPE training_status AS ENUM ('Scheduled', 'Pending', 'Success', 'Fail');

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Core Thrift Entity Tables

-- Table for Budget entity
CREATE TABLE budgets (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    last_modified_on TIMESTAMP WITH TIME ZONE,
    first_month DATE,
    last_month DATE,
    date_format_id VARCHAR(255),
    currency_format_id VARCHAR(255),
    platform_type budgeting_platform_type DEFAULT 'YNAB',
    currency VARCHAR(10) DEFAULT 'USD',
    total_amount DOUBLE PRECISION,
    start_date VARCHAR(50),
    end_date VARCHAR(50),
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for Account entity
CREATE TABLE accounts (
    id UUID PRIMARY KEY,
    budget_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(255),
    on_budget BOOLEAN,
    closed BOOLEAN,
    note TEXT,
    balance BIGINT, -- In milliunits
    cleared_balance BIGINT,
    uncleared_balance BIGINT,
    transfer_payee_id UUID,
    deleted BOOLEAN,
    platform_type budgeting_platform_type DEFAULT 'YNAB',
    institution VARCHAR(255),
    currency VARCHAR(10) DEFAULT 'USD',
    status VARCHAR(50) DEFAULT 'active',
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (budget_id) REFERENCES budgets(id)
);

-- Table for Category entity
CREATE TABLE categories (
    id UUID PRIMARY KEY,
    budget_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    hidden BOOLEAN,
    budgeted BIGINT, -- In milliunits
    activity BIGINT, -- In milliunits
    balance BIGINT, -- In milliunits
    note TEXT,
    deleted BOOLEAN,
    platform_type budgeting_platform_type DEFAULT 'YNAB',
    description TEXT,
    is_income_category BOOLEAN DEFAULT FALSE,
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (budget_id) REFERENCES budgets(id)
);

-- Table for Payee entity
CREATE TABLE payees (
    id UUID PRIMARY KEY,
    budget_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    transfer_account_id UUID,
    deleted BOOLEAN,
    platform_type budgeting_platform_type DEFAULT 'YNAB',
    description TEXT,
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (budget_id) REFERENCES budgets(id)
);

-- Table for Transaction entity
CREATE TABLE transactions (
    id UUID PRIMARY KEY,
    budget_id UUID NOT NULL,
    account_id UUID NOT NULL,
    date DATE,
    amount BIGINT NOT NULL, -- In milliunits
    memo TEXT,
    cleared VARCHAR(255),
    approved BOOLEAN,
    flag_color VARCHAR(255),
    account_name VARCHAR(255),
    payee_id UUID,
    payee_name VARCHAR(255),
    category_id UUID,
    category_name VARCHAR(255),
    transfer_account_id UUID,
    transfer_transaction_id UUID,
    matched_transaction_id UUID,
    import_id VARCHAR(255),
    deleted BOOLEAN,
    parent_transaction_id UUID,
    is_parent_transaction BOOLEAN DEFAULT FALSE NOT NULL,
    is_sub_transaction BOOLEAN DEFAULT FALSE NOT NULL,
    schedule_date_first DATE,
    schedule_date_next DATE,
    schedule_frequency VARCHAR(255),
    platform_type budgeting_platform_type DEFAULT 'YNAB',
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (budget_id) REFERENCES budgets(id),
    FOREIGN KEY (account_id) REFERENCES accounts(id),
    FOREIGN KEY (payee_id) REFERENCES payees(id),
    FOREIGN KEY (category_id) REFERENCES categories(id),
    FOREIGN KEY (transfer_account_id) REFERENCES accounts(id),
    FOREIGN KEY (parent_transaction_id) REFERENCES transactions(id)
);

-- Table for Metadata entity
CREATE TABLE metadata (
    id VARCHAR(255) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    transaction_id UUID NOT NULL,
    type metadata_type NOT NULL,
    value JSONB NOT NULL,
    source_system JSONB NOT NULL,
    description TEXT,
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (transaction_id) REFERENCES transactions(id) ON DELETE CASCADE
);

-- Table for ConfigItem entity
CREATE TABLE config_items (
    key VARCHAR(255) PRIMARY KEY,
    type config_type NOT NULL,
    value JSONB NOT NULL,
    description TEXT,
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for ModelInfo entity
CREATE TABLE model_info (
    name VARCHAR(255) PRIMARY KEY,
    model_type model_type NOT NULL,
    version VARCHAR(100) NOT NULL,
    description TEXT,
    status training_status NOT NULL,
    trained_date VARCHAR(50),
    performance_metrics JSONB,
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for FileEntity entity
CREATE TABLE file_entities (
    path VARCHAR(500) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    content_type VARCHAR(100),
    size BIGINT,
    checksum VARCHAR(255),
    last_modified BIGINT,
    metadata JSONB,
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for history tracking (not a thrift entity but needed for audit)
CREATE TABLE history_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type entity_type NOT NULL,
    entity_id UUID NOT NULL,
    field_name VARCHAR(255) NOT NULL,
    old_value JSONB,
    new_value JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_action VARCHAR(255) NOT NULL,
    can_undo BOOLEAN DEFAULT TRUE,
    undone BOOLEAN DEFAULT FALSE,
    undo_timestamp TIMESTAMP WITH TIME ZONE,
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create triggers for updated_at on all tables
CREATE TRIGGER update_budgets_updated_at BEFORE UPDATE ON budgets FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_accounts_updated_at BEFORE UPDATE ON accounts FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_categories_updated_at BEFORE UPDATE ON categories FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_payees_updated_at BEFORE UPDATE ON payees FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_transactions_updated_at BEFORE UPDATE ON transactions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_metadata_updated_at BEFORE UPDATE ON metadata FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_config_items_updated_at BEFORE UPDATE ON config_items FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_model_info_updated_at BEFORE UPDATE ON model_info FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_file_entities_updated_at BEFORE UPDATE ON file_entities FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_history_entries_updated_at BEFORE UPDATE ON history_entries FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create indexes for performance

-- Budget indexes
CREATE INDEX idx_budgets_name ON budgets(name);
CREATE INDEX idx_budgets_platform_type ON budgets(platform_type);

-- Account indexes
CREATE INDEX idx_accounts_budget_id ON accounts(budget_id);
CREATE INDEX idx_accounts_name ON accounts(name);
CREATE INDEX idx_accounts_type ON accounts(type);
CREATE INDEX idx_accounts_closed ON accounts(closed);
CREATE INDEX idx_accounts_deleted ON accounts(deleted);

-- Category indexes
CREATE INDEX idx_categories_budget_id ON categories(budget_id);
CREATE INDEX idx_categories_name ON categories(name);
CREATE INDEX idx_categories_deleted ON categories(deleted);

-- Payee indexes
CREATE INDEX idx_payees_budget_id ON payees(budget_id);
CREATE INDEX idx_payees_name ON payees(name);
CREATE INDEX idx_payees_deleted ON payees(deleted);

-- Transaction indexes
CREATE INDEX idx_transactions_budget_id ON transactions(budget_id);
CREATE INDEX idx_transactions_account_id ON transactions(account_id);
CREATE INDEX idx_transactions_payee_id ON transactions(payee_id);
CREATE INDEX idx_transactions_category_id ON transactions(category_id);
CREATE INDEX idx_transactions_date ON transactions(date);
CREATE INDEX idx_transactions_deleted ON transactions(deleted);
CREATE INDEX idx_transactions_parent_id ON transactions(parent_transaction_id);
CREATE INDEX idx_transactions_approved ON transactions(approved);

-- Metadata indexes
CREATE INDEX idx_metadata_type ON metadata(type);

-- Config items indexes
CREATE INDEX idx_config_items_type ON config_items(type);

-- Model info indexes
CREATE INDEX idx_model_info_type ON model_info(model_type);
CREATE INDEX idx_model_info_status ON model_info(status);

-- File entities indexes
CREATE INDEX idx_file_entities_content_type ON file_entities(content_type);
CREATE INDEX idx_file_entities_size ON file_entities(size);

-- History indexes
CREATE INDEX idx_history_entity ON history_entries(entity_type, entity_id);
CREATE INDEX idx_history_timestamp ON history_entries(timestamp);

-- Add foreign key constraints
ALTER TABLE accounts 
ADD CONSTRAINT fk_accounts_transfer_payee 
FOREIGN KEY (transfer_payee_id) REFERENCES payees(id);

ALTER TABLE payees 
ADD CONSTRAINT fk_payees_transfer_account 
FOREIGN KEY (transfer_account_id) REFERENCES accounts(id);

-- migrate:down

-- Drop foreign key constraints
ALTER TABLE accounts DROP CONSTRAINT IF EXISTS fk_accounts_transfer_payee;
ALTER TABLE payees DROP CONSTRAINT IF EXISTS fk_payees_transfer_account;

-- Drop triggers
DROP TRIGGER IF EXISTS update_budgets_updated_at ON budgets;
DROP TRIGGER IF EXISTS update_accounts_updated_at ON accounts;
DROP TRIGGER IF EXISTS update_categories_updated_at ON categories;
DROP TRIGGER IF EXISTS update_payees_updated_at ON payees;
DROP TRIGGER IF EXISTS update_transactions_updated_at ON transactions;
DROP TRIGGER IF EXISTS update_metadata_updated_at ON metadata;
DROP TRIGGER IF EXISTS update_config_items_updated_at ON config_items;
DROP TRIGGER IF EXISTS update_model_info_updated_at ON model_info;
DROP TRIGGER IF EXISTS update_file_entities_updated_at ON file_entities;
DROP TRIGGER IF EXISTS update_history_entries_updated_at ON history_entries;

-- Drop function
DROP FUNCTION IF EXISTS update_updated_at_column();

-- Drop tables in reverse order
DROP TABLE IF EXISTS history_entries;
DROP TABLE IF EXISTS file_entities;
DROP TABLE IF EXISTS model_info;
DROP TABLE IF EXISTS config_items;
DROP TABLE IF EXISTS metadata;
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS payees;
DROP TABLE IF EXISTS categories;
DROP TABLE IF EXISTS accounts;
DROP TABLE IF EXISTS budgets;

-- Drop enum types
DROP TYPE IF EXISTS training_status;
DROP TYPE IF EXISTS sync_status;
DROP TYPE IF EXISTS config_type;
DROP TYPE IF EXISTS model_type;
DROP TYPE IF EXISTS email_platform_type;
DROP TYPE IF EXISTS metadata_type;
DROP TYPE IF EXISTS budgeting_platform_type;
DROP TYPE IF EXISTS entity_type;