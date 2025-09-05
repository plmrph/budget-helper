---
inclusion: always
---

# Task Start Checklist

Before beginning any task, ALWAYS:

## 1. Understand the Architecture
- **Read the context**: What layer am I working in? (API, Business, Resource Access, Resource)
- **Check dependencies**: What services/components does this interact with?
- **Verify Thrift contracts**: Are there existing service interfaces I need to follow?
- **Review The Method rules**: Am I allowed to call the components I need?

## 2. Plan the Approach
- **Identify the minimal change**: What's the smallest modification that solves the problem?
- **Check existing patterns**: How do similar components handle this?
- **Consider reusability**: Can I use existing components instead of creating new ones?
- **Think about testing**: How will I verify this works?

## 3. Technical Preparation
- **Use generated Thrift types**: Import from `backend/thrift_gen/*/ttypes.py`, don't create new types
- **Follow container-first development**: Plan to run/test through Docker containers
- **Check file locations**: Am I modifying the right files according to the architecture?
- **Verify I'm not modifying `.llm/core/`**: Those files are immutable without approval

## 4. Code Quality Setup
- **Plan import organization**: All imports go at the top of files
- **Follow naming conventions**: Private methods start with `_`, public methods match Thrift interfaces
- **Consider error handling**: What exceptions should I throw according to Thrift contracts?
- **Plan for cleanup**: Will I create any temporary files that need removal?

## 5. Scope Verification & Requirements Gathering
- **Understand the user's actual request**: What problem are they trying to solve?
- **Identify the minimum viable solution**: Don't over-engineer
- **Check for existing solutions**: Has this been solved elsewhere in the codebase?
- **Clarify ambiguities**: Ask questions if the requirements aren't clear

### Deep Requirements Understanding
Before starting implementation, ask ALL clarifying questions upfront:

- **What's the complete user workflow?** How does this fit into the bigger picture?
- **What are the edge cases?** What happens when things go wrong?
- **What are the performance requirements?** How much data? How fast should it be?
- **What's the user experience expectation?** Should this be instant? Can it be async?
- **Are there dependencies I'm missing?** What other systems/components are involved?
- **What's the data flow?** Where does data come from and where does it go?
- **What are the validation rules?** What makes input valid/invalid?
- **What's the error handling strategy?** How should failures be communicated?
- **Are there security considerations?** Who can access this? What data is sensitive?
- **What's the testing strategy?** How will we know this works correctly?
- **What's the rollback plan?** How do we undo this if something goes wrong?
- **Are there configuration needs?** Should this be configurable? By whom?

### Get the Full Picture First
- **Ask about the end-to-end flow**: Don't just solve the immediate problem, understand the complete user journey
- **Identify all stakeholders**: Who else will be affected by this change?
- **Understand the timeline**: Is this urgent? Can we do it in phases?
- **Check for hidden requirements**: What assumptions am I making that might be wrong?
- **Verify success criteria**: How will we know this is working correctly?

**RULE: If you have ANY uncertainty about requirements, ask ALL your questions in one go before starting implementation. Don't implement and then ask questions.**