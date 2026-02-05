# Project Requirements

## Overview

This project is a self-improving data analysis agent that learns from its mistakes and improves over time. The agent demonstrates continuous learning across sessions, not just within a single conversation.

## The Challenge

Build a **self-improving data analysis chatbot** that:

1. **Analyzes tabular data** (CSV files) through natural language questions
2. **Detects when it makes mistakes** during analysis
3. **Learns from those mistakes** by creating persistent improvements
4. **Demonstrates meta-learning**: The agent gets better **across sessions** - not just within a single conversation

### Example Workflow

**Session 1:**

- User asks: "What was the total Q1 revenue for Product A in 2024?"
- Agent attempts to answer but makes an error (e.g., wrong date filtering logic)
- Agent detects the failure, analyzes what went wrong
- Agent creates a persistent improvement (new helper function, validation rule, or knowledge entry)

**Session 2 (fresh conversation, days later):**

- Different user asks: "What was Q2 revenue for Product B in 2023?"
- Agent successfully applies the learned date filtering logic from Session 1
- No repeat of the previous mistake - the agent has genuinely improved

## Key Functional Requirements

### Core Features (Required)

**1. Natural Language → Data Analysis**

- Accept natural language questions about tabular data
- Generate and execute code to answer questions
- Return results in a clear, structured format

**2. Error Detection**

- Detect when analysis produces incorrect or invalid results
- Identify the type of error (logic error, data misunderstanding, edge case, etc.)

**3. Persistent Improvement Mechanism**

- Analyze failures to determine root cause
- Generate improvements (helper functions, validations, corrections, knowledge)
- **Persist improvements across sessions** using external storage
- Apply learned improvements to future queries in **new sessions**

**4. Code Execution**

- Implement safe code execution for generated Python analysis
- Choose appropriate approach: subprocess sandboxing, restricted Python, or other methods
- Justify the decision in the design document

**5. Demo & Documentation**

- Working demo showing cross-session learning
- Clear explanation of improvement storage and retrieval architecture

### Stretch Goals (Optional)

- **Data Visualization**: Generate charts/graphs for data insights
- **Multi-CSV Support**: Analyze across multiple related datasets
- **Confidence Scoring**: Assess confidence in answers and request validation when uncertain
- **Improvement Versioning**: Track evolution of learned knowledge over time
- **Rollback Mechanism**: Ability to undo ineffective improvements

## Technical Considerations

### General Architecture

- **Agent Framework**: Structure the agentic loop (prompt → tool calls → execution → feedback)
- **Code Execution Strategy**: Safely execute generated Python code
- **Error Detection**: Define signals that indicate mistakes and classify errors
- **State Management**: Persist learned knowledge across sessions

### Continuous Learning Design (The Core Challenge)

This is the heart of the project - designing a system where **Session N+1 is measurably better than Session N**.

**Key questions to address:**

**Storage Mechanism**: How do you persist improvements?

- Git commits to a repository?
- External database or knowledge base?
- File-based storage (JSON, Python modules)?
- Vector store for semantic retrieval?
- Redis-like persistent state?

**Improvement Types**: What can the agent learn?

- Helper functions that get added to its toolkit?
- Validation rules that prevent known errors?
- Domain knowledge about data patterns?
- Prompt modifications or system instructions?
- Code templates for common patterns?

**Retrieval & Application**: How are improvements applied in new sessions?

- Loaded automatically at agent initialization?
- Retrieved based on query similarity?
- Applied through modified prompts?
- Injected as available tools?

**Evaluation**: How do you measure whether improvements are effective?

### Implementation Stack

**Required:**

- **Python 3.11+**
- **Agentic SDK with tool calling support**

**Recommended libraries**:

- `pandas` for data analysis
- Standard library tools for chosen approach

**You choose:**

- Code execution method
- Storage mechanism for improvements
- Any additional libraries that support the architecture

## Deliverables

### 1. Working Implementation

A Python codebase that demonstrates the self-improving agent. Should include:

- **Main agent code** (agentic loop, tool definitions, improvement logic)
- **Persistent storage mechanism** (chosen approach)
- **Setup instructions** (README with clear steps to run)
- **Test scenarios** that demonstrate cross-session learning

### 2. Demo

Demonstrate the agent working:

1. **Initial state**: Agent makes a mistake on Question A
2. **Learning**: Agent analyzes the mistake and creates a persistent improvement
3. **Fresh session**: Start a completely new conversation (simulating days later)
4. **Improved state**: Agent handles Question B (similar pattern to A) correctly
5. **Evidence**: Show the stored improvement (file diff, database entry, etc.)

### 3. Design Document

A document explaining:

**Architecture Overview:**

- System diagram showing components and data flow
- How the agentic loop works
- Where improvements are stored and how they're retrieved

**Self-Improvement Mechanism:**

- What triggers improvement creation?
- How are improvements represented? (code? data? prompts?)
- Where are they stored? (justify choice)
- How are they applied in new sessions?

**Code Execution Strategy:**

- What approach was chosen and why?
- How is safety/sandboxing ensured?
- Trade-offs of the approach

**Evaluation Strategy:**

- How is improvement effectiveness measured?
- How would bad improvements be prevented from persisting?

**Production Considerations:**

- What would need to change for a production system?
- Scalability concerns
- Security considerations

### 4. Code Repository

- Clean commit history showing development process
- Clear README with setup and running instructions

## Sample Dataset

## Dataset: `FUN_company_pl_actuals_dataset.csv`

**Specifications:**

- 📏 **21,601 rows** of financial data
- 📅 **5 years**: 2020-2024
- 📆 **20 quarters** of data
- 📦 **4 products**: Product A, B, C, D
- 🌍 **6 countries**: Australia, Canada, Germany, Japan, United Kingdom, United States
- 💱 **Multi-currency**: AUD, CAD, EUR, GBP, JPY, USD (with USD conversion)

**Columns:**

1. Fiscal Year
2. Fiscal Quarter
3. Fiscal Period (YYYY-MM format)
4. FSLine Statement L1 (High-level category)
5. FSLine Statement L2 (Detailed line item)
6. Product
7. Country
8. Currency
9. Amount in Local Currency
10. Amount in USD
11. Version (all rows are 'Actuals')

**Financial Statement Structure:**

**Level 1 Categories:**

- **Net Revenue**: Total sales and revenue
- **Cost of Goods Sold (COGS)**: Direct costs of production
- **OPEX**: Operating expenses
- **Other Income/Expenses**: Non-operating items

**Level 2 Line Items** (15 detailed categories):

- Revenue: Gross Revenue, Returns and Refunds, Revenue Adjustment
- COGS: Direct Labor, Direct Materials, Manufacturing Overhead
- OPEX: Marketing Expenses, R&D Expenses, Sales Expenses, General & Administrative, IT Expenses, Headcount Expenses
- Other: Interest Income, Interest Expense, Foreign Exchange Gain/Loss

**Sample Data Preview:**

```
Fiscal Year,Quarter,Period,L1,L2,Product,Country,Currency,Local,USD,Version
2020,Q1,2020-01,Net Revenue,Gross Revenue,Product A,Australia,AUD,213437.77,149406.44,Actuals
2020,Q1,2020-01,Net Revenue,Returns and Refunds,Product A,Australia,AUD,-8080.32,-5656.22,Actuals
2020,Q1,2020-01,Cost of Goods Sold,Direct Labor,Product A,Australia,AUD,19182.39,13427.67,Actuals
2020,Q1,2020-01,OPEX,Marketing Expenses,Product A,Australia,AUD,21457.59,15020.31,Actuals
```

## Sample Test Questions

### Easy (Basic Filtering)

**Q:** What was the Gross Revenue for Product A in the United States in Q1 2020?
**Tests:** Basic filtering by product, country, quarter, year, and financial line item

**Q:** How much did the company spend on Marketing Expenses globally in Q2 2023?
**Tests:** Filtering and aggregation across all countries

### Medium (Multi-Step Aggregation)

**Q:** Calculate the total Net Revenue for all products in Q4 2023
**Tests:** Understanding that Net Revenue = Gross Revenue + Returns + Revenue Adjustments (with negative values)

**Q:** What was the year-over-year growth in total OPEX between Q1 2022 and Q1 2023?
**Tests:** Multi-quarter comparison, percentage calculations, aggregating all OPEX subcategories

### Hard (Complex Analysis)

**Q:** Which product had the highest operating margin in Q3 2023?
**Tests:** Calculating Operating Margin = (Revenue - COGS - OPEX) / Revenue, comparing across products

**Q:** What was the foreign exchange impact for Product C across all countries in 2024?
**Tests:** Filtering specific line items, aggregating across quarters/countries, handling negative values

### Very Hard (Edge Cases)

**Q:** Compare the Cost of Goods Sold as a percentage of Gross Revenue between 2020 and 2024 for Product B
**Tests:** Multi-year aggregation, percentage calculations, understanding financial relationships

### Trick Questions (Error Detection)

**Q:** What was the total revenue for Product E in Q1 2023?
**Expected:** "Product E does not exist" (only A, B, C, D exist)

**Q:** Calculate the Employee Headcount in Japan for Q2 2024
**Expected:** "Employee Headcount not in dataset" (only Headcount Expenses exists)
