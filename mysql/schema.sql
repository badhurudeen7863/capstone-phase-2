-- ============================================================================
-- ExpenseIQ - MySQL 8 schema (normalised, 3NF) - project report section 3.3
-- Usage:
--   mysql -u root -p < mysql/schema.sql
--   then set in .env:
--   DATABASE_URL=mysql+pymysql://root:your_password@localhost:3306/expenseiq
-- (SQLAlchemy can also create these tables automatically; this script is the
--  canonical DDL for DBAs / viva demonstrations.)
-- ============================================================================

CREATE DATABASE IF NOT EXISTS expenseiq
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE expenseiq;

CREATE TABLE IF NOT EXISTS users (
    user_id       INT AUTO_INCREMENT PRIMARY KEY,
    name          VARCHAR(120)  NOT NULL,
    email         VARCHAR(190)  NOT NULL UNIQUE,
    password_hash VARCHAR(255)  NOT NULL,               -- bcrypt
    created_at    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS categories (
    category_id INT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(80) NOT NULL UNIQUE,
    type        ENUM('Fixed','Variable') NOT NULL DEFAULT 'Variable'
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS transactions (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    user_id          INT           NOT NULL,
    category_id      INT           NOT NULL,
    amount           DECIMAL(10,2) NOT NULL,
    transaction_date DATE          NOT NULL,
    notes            TEXT          NULL,
    created_at       DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_txn_user     FOREIGN KEY (user_id)     REFERENCES users(user_id)      ON DELETE CASCADE,
    CONSTRAINT fk_txn_category FOREIGN KEY (category_id) REFERENCES categories(category_id),
    CONSTRAINT ck_txn_amount   CHECK (amount > 0),
    INDEX ix_txn_user_date (user_id, transaction_date),
    INDEX ix_txn_date (transaction_date)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS predictions_log (
    prediction_id    INT AUTO_INCREMENT PRIMARY KEY,
    user_id          INT      NOT NULL,
    forecast_month   DATE     NOT NULL,              -- 1st day of target month
    predicted_amount DOUBLE   NOT NULL,
    lower_bound      DOUBLE   NOT NULL,
    upper_bound      DOUBLE   NOT NULL,
    confidence_level DOUBLE   NOT NULL DEFAULT 0.95,
    model_version    VARCHAR(80) NULL,
    actual_val       DOUBLE   NULL,                  -- filled once month ends (drift audit)
    created_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_pred_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    INDEX ix_pred_user_month (user_id, forecast_month)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS budgets (
    budget_id   INT AUTO_INCREMENT PRIMARY KEY,
    user_id     INT           NOT NULL,
    category_id INT           NOT NULL,
    month       DATE          NOT NULL,              -- 1st day of month
    amount      DECIMAL(10,2) NOT NULL,
    CONSTRAINT fk_budget_user     FOREIGN KEY (user_id)     REFERENCES users(user_id) ON DELETE CASCADE,
    CONSTRAINT fk_budget_category FOREIGN KEY (category_id) REFERENCES categories(category_id),
    CONSTRAINT ck_budget_amount   CHECK (amount >= 0),
    CONSTRAINT uq_budget_user_cat_month UNIQUE (user_id, category_id, month)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------- seed data --
INSERT INTO categories (name, type) VALUES
    ('Rent',             'Fixed'),
    ('Utilities',        'Fixed'),
    ('Food & Groceries', 'Variable'),
    ('Health',           'Variable'),
    ('Transport',        'Variable'),
    ('Entertainment',    'Variable'),
    ('Other',            'Variable')
ON DUPLICATE KEY UPDATE name = VALUES(name);
