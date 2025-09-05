SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: budgeting_platform_type; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.budgeting_platform_type AS ENUM (
    'YNAB'
);


--
-- Name: config_type; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.config_type AS ENUM (
    'System',
    'Email',
    'AI',
    'Display',
    'ExternalSystem'
);


--
-- Name: email_platform_type; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.email_platform_type AS ENUM (
    'Gmail'
);


--
-- Name: entity_type; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.entity_type AS ENUM (
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


--
-- Name: metadata_type; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.metadata_type AS ENUM (
    'Email',
    'Prediction'
);


--
-- Name: model_type; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.model_type AS ENUM (
    'PXBlendSC'
);


--
-- Name: sync_status; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.sync_status AS ENUM (
    'Success',
    'Pending',
    'Partial',
    'Fail'
);


--
-- Name: training_status; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.training_status AS ENUM (
    'Scheduled',
    'Pending',
    'Success',
    'Fail'
);


--
-- Name: update_updated_at_column(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.update_updated_at_column() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: accounts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.accounts (
    id uuid NOT NULL,
    budget_id uuid NOT NULL,
    name character varying(255) NOT NULL,
    type character varying(255),
    on_budget boolean,
    closed boolean,
    note text,
    balance bigint,
    cleared_balance bigint,
    uncleared_balance bigint,
    transfer_payee_id uuid,
    deleted boolean,
    platform_type public.budgeting_platform_type DEFAULT 'YNAB'::public.budgeting_platform_type,
    institution character varying(255),
    currency character varying(10) DEFAULT 'USD'::character varying,
    status character varying(50) DEFAULT 'active'::character varying,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: budgets; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.budgets (
    id uuid NOT NULL,
    name character varying(255) NOT NULL,
    last_modified_on timestamp with time zone,
    first_month date,
    last_month date,
    date_format_id character varying(255),
    currency_format_id character varying(255),
    platform_type public.budgeting_platform_type DEFAULT 'YNAB'::public.budgeting_platform_type,
    currency character varying(10) DEFAULT 'USD'::character varying,
    total_amount double precision,
    start_date character varying(50),
    end_date character varying(50),
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: categories; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.categories (
    id uuid NOT NULL,
    budget_id uuid NOT NULL,
    name character varying(255) NOT NULL,
    hidden boolean,
    budgeted bigint,
    activity bigint,
    balance bigint,
    note text,
    deleted boolean,
    platform_type public.budgeting_platform_type DEFAULT 'YNAB'::public.budgeting_platform_type,
    description text,
    is_income_category boolean DEFAULT false,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: config_items; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.config_items (
    key character varying(255) NOT NULL,
    type public.config_type NOT NULL,
    value jsonb NOT NULL,
    description text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: file_entities; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.file_entities (
    path character varying(500) NOT NULL,
    name character varying(255) NOT NULL,
    content_type character varying(100),
    size bigint,
    checksum character varying(255),
    last_modified bigint,
    metadata jsonb,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: history_entries; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.history_entries (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    entity_type public.entity_type NOT NULL,
    entity_id uuid NOT NULL,
    field_name character varying(255) NOT NULL,
    old_value jsonb,
    new_value jsonb,
    "timestamp" timestamp with time zone DEFAULT now(),
    user_action character varying(255) NOT NULL,
    can_undo boolean DEFAULT true,
    undone boolean DEFAULT false,
    undo_timestamp timestamp with time zone,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: metadata; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.metadata (
    id character varying(255) DEFAULT (gen_random_uuid())::text NOT NULL,
    transaction_id uuid NOT NULL,
    type public.metadata_type NOT NULL,
    value jsonb NOT NULL,
    source_system jsonb NOT NULL,
    description text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: model_info; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.model_info (
    name character varying(255) NOT NULL,
    model_type public.model_type NOT NULL,
    version character varying(100) NOT NULL,
    description text,
    status public.training_status NOT NULL,
    trained_date character varying(50),
    performance_metrics jsonb,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: payees; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.payees (
    id uuid NOT NULL,
    budget_id uuid NOT NULL,
    name character varying(255) NOT NULL,
    transfer_account_id uuid,
    deleted boolean,
    platform_type public.budgeting_platform_type DEFAULT 'YNAB'::public.budgeting_platform_type,
    description text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: schema_migrations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.schema_migrations (
    version character varying NOT NULL
);


--
-- Name: transactions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.transactions (
    id uuid NOT NULL,
    budget_id uuid NOT NULL,
    account_id uuid NOT NULL,
    date date,
    amount bigint NOT NULL,
    memo text,
    cleared character varying(255),
    approved boolean,
    flag_color character varying(255),
    account_name character varying(255),
    payee_id uuid,
    payee_name character varying(255),
    category_id uuid,
    category_name character varying(255),
    transfer_account_id uuid,
    transfer_transaction_id uuid,
    matched_transaction_id uuid,
    import_id character varying(255),
    deleted boolean,
    parent_transaction_id uuid,
    is_parent_transaction boolean DEFAULT false NOT NULL,
    is_sub_transaction boolean DEFAULT false NOT NULL,
    schedule_date_first date,
    schedule_date_next date,
    schedule_frequency character varying(255),
    platform_type public.budgeting_platform_type DEFAULT 'YNAB'::public.budgeting_platform_type,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: accounts accounts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.accounts
    ADD CONSTRAINT accounts_pkey PRIMARY KEY (id);


--
-- Name: budgets budgets_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.budgets
    ADD CONSTRAINT budgets_pkey PRIMARY KEY (id);


--
-- Name: categories categories_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.categories
    ADD CONSTRAINT categories_pkey PRIMARY KEY (id);


--
-- Name: config_items config_items_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.config_items
    ADD CONSTRAINT config_items_pkey PRIMARY KEY (key);


--
-- Name: file_entities file_entities_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.file_entities
    ADD CONSTRAINT file_entities_pkey PRIMARY KEY (path);


--
-- Name: history_entries history_entries_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.history_entries
    ADD CONSTRAINT history_entries_pkey PRIMARY KEY (id);


--
-- Name: metadata metadata_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.metadata
    ADD CONSTRAINT metadata_pkey PRIMARY KEY (id);


--
-- Name: model_info model_info_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_info
    ADD CONSTRAINT model_info_pkey PRIMARY KEY (name);


--
-- Name: payees payees_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.payees
    ADD CONSTRAINT payees_pkey PRIMARY KEY (id);


--
-- Name: schema_migrations schema_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.schema_migrations
    ADD CONSTRAINT schema_migrations_pkey PRIMARY KEY (version);


--
-- Name: transactions transactions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.transactions
    ADD CONSTRAINT transactions_pkey PRIMARY KEY (id);


--
-- Name: idx_accounts_budget_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_accounts_budget_id ON public.accounts USING btree (budget_id);


--
-- Name: idx_accounts_closed; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_accounts_closed ON public.accounts USING btree (closed);


--
-- Name: idx_accounts_deleted; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_accounts_deleted ON public.accounts USING btree (deleted);


--
-- Name: idx_accounts_name; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_accounts_name ON public.accounts USING btree (name);


--
-- Name: idx_accounts_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_accounts_type ON public.accounts USING btree (type);


--
-- Name: idx_budgets_name; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_budgets_name ON public.budgets USING btree (name);


--
-- Name: idx_budgets_platform_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_budgets_platform_type ON public.budgets USING btree (platform_type);


--
-- Name: idx_categories_budget_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_categories_budget_id ON public.categories USING btree (budget_id);


--
-- Name: idx_categories_deleted; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_categories_deleted ON public.categories USING btree (deleted);


--
-- Name: idx_categories_name; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_categories_name ON public.categories USING btree (name);


--
-- Name: idx_config_items_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_config_items_type ON public.config_items USING btree (type);


--
-- Name: idx_file_entities_content_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_file_entities_content_type ON public.file_entities USING btree (content_type);


--
-- Name: idx_file_entities_size; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_file_entities_size ON public.file_entities USING btree (size);


--
-- Name: idx_history_entity; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_history_entity ON public.history_entries USING btree (entity_type, entity_id);


--
-- Name: idx_history_timestamp; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_history_timestamp ON public.history_entries USING btree ("timestamp");


--
-- Name: idx_metadata_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_metadata_type ON public.metadata USING btree (type);


--
-- Name: idx_model_info_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_model_info_status ON public.model_info USING btree (status);


--
-- Name: idx_model_info_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_model_info_type ON public.model_info USING btree (model_type);


--
-- Name: idx_payees_budget_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_payees_budget_id ON public.payees USING btree (budget_id);


--
-- Name: idx_payees_deleted; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_payees_deleted ON public.payees USING btree (deleted);


--
-- Name: idx_payees_name; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_payees_name ON public.payees USING btree (name);


--
-- Name: idx_transactions_account_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_transactions_account_id ON public.transactions USING btree (account_id);


--
-- Name: idx_transactions_approved; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_transactions_approved ON public.transactions USING btree (approved);


--
-- Name: idx_transactions_budget_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_transactions_budget_id ON public.transactions USING btree (budget_id);


--
-- Name: idx_transactions_category_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_transactions_category_id ON public.transactions USING btree (category_id);


--
-- Name: idx_transactions_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_transactions_date ON public.transactions USING btree (date);


--
-- Name: idx_transactions_deleted; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_transactions_deleted ON public.transactions USING btree (deleted);


--
-- Name: idx_transactions_parent_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_transactions_parent_id ON public.transactions USING btree (parent_transaction_id);


--
-- Name: idx_transactions_payee_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_transactions_payee_id ON public.transactions USING btree (payee_id);


--
-- Name: accounts update_accounts_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER update_accounts_updated_at BEFORE UPDATE ON public.accounts FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: budgets update_budgets_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER update_budgets_updated_at BEFORE UPDATE ON public.budgets FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: categories update_categories_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER update_categories_updated_at BEFORE UPDATE ON public.categories FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: config_items update_config_items_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER update_config_items_updated_at BEFORE UPDATE ON public.config_items FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: file_entities update_file_entities_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER update_file_entities_updated_at BEFORE UPDATE ON public.file_entities FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: history_entries update_history_entries_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER update_history_entries_updated_at BEFORE UPDATE ON public.history_entries FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: metadata update_metadata_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER update_metadata_updated_at BEFORE UPDATE ON public.metadata FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: model_info update_model_info_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER update_model_info_updated_at BEFORE UPDATE ON public.model_info FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: payees update_payees_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER update_payees_updated_at BEFORE UPDATE ON public.payees FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: transactions update_transactions_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER update_transactions_updated_at BEFORE UPDATE ON public.transactions FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: accounts accounts_budget_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.accounts
    ADD CONSTRAINT accounts_budget_id_fkey FOREIGN KEY (budget_id) REFERENCES public.budgets(id);


--
-- Name: categories categories_budget_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.categories
    ADD CONSTRAINT categories_budget_id_fkey FOREIGN KEY (budget_id) REFERENCES public.budgets(id);


--
-- Name: accounts fk_accounts_transfer_payee; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.accounts
    ADD CONSTRAINT fk_accounts_transfer_payee FOREIGN KEY (transfer_payee_id) REFERENCES public.payees(id);


--
-- Name: payees fk_payees_transfer_account; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.payees
    ADD CONSTRAINT fk_payees_transfer_account FOREIGN KEY (transfer_account_id) REFERENCES public.accounts(id);


--
-- Name: metadata metadata_transaction_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.metadata
    ADD CONSTRAINT metadata_transaction_id_fkey FOREIGN KEY (transaction_id) REFERENCES public.transactions(id) ON DELETE CASCADE;


--
-- Name: payees payees_budget_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.payees
    ADD CONSTRAINT payees_budget_id_fkey FOREIGN KEY (budget_id) REFERENCES public.budgets(id);


--
-- Name: transactions transactions_account_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.transactions
    ADD CONSTRAINT transactions_account_id_fkey FOREIGN KEY (account_id) REFERENCES public.accounts(id);


--
-- Name: transactions transactions_budget_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.transactions
    ADD CONSTRAINT transactions_budget_id_fkey FOREIGN KEY (budget_id) REFERENCES public.budgets(id);


--
-- Name: transactions transactions_category_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.transactions
    ADD CONSTRAINT transactions_category_id_fkey FOREIGN KEY (category_id) REFERENCES public.categories(id);


--
-- Name: transactions transactions_parent_transaction_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.transactions
    ADD CONSTRAINT transactions_parent_transaction_id_fkey FOREIGN KEY (parent_transaction_id) REFERENCES public.transactions(id);


--
-- Name: transactions transactions_payee_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.transactions
    ADD CONSTRAINT transactions_payee_id_fkey FOREIGN KEY (payee_id) REFERENCES public.payees(id);


--
-- Name: transactions transactions_transfer_account_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.transactions
    ADD CONSTRAINT transactions_transfer_account_id_fkey FOREIGN KEY (transfer_account_id) REFERENCES public.accounts(id);


--
-- PostgreSQL database dump complete
--


--
-- Dbmate schema migrations
--

INSERT INTO public.schema_migrations (version) VALUES
    ('20250104000001');
