-- Script to create a table called "users" in the current database in your MySQL server.
CREATE TABLE IF NOT EXISTS users (
    id INT NOT NULL AUTO_Increment,
    email, VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(256),
    country ENUM('US', 'CO', 'TN') DEFAULT 'US' NOT NULL,
    PRIMARY KEY (id)
);
