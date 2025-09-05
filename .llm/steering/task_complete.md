---
inclusion: always
---

# Task Completion Checklist

Before marking any task as complete, ALWAYS verify:

## 1. Code Quality
- **Import Placement**: All imports are at the top of Python files, never in the middle
- **PEP 8 Compliance**: Code follows Python style guidelines
- **Ruff Linting**: Run `uv run ruff check . --fix` inside the "backend" folder to check for linting issues
- **Ruff Formatting**: Run `uv run ruff format .` inside the "backend" folder  to format all Python files
- **No Useless Comments**: Removed self-evident comments like "# moved to..." or "# PUBLIC METHODS"
- **Proper Error Handling**: Using Thrift-defined exceptions appropriately

## 2. Architecture Compliance
- **The Method Rules**: No forbidden layer interactions (up-calls, sideways calls)
- **Thrift Types**: Using generated types from `thrift_gen/*/ttypes.py`, not creating new ones
- **Service Contracts**: Public methods match Thrift interface definitions exactly
- **Layer Responsibilities**: Changes are appropriate for the layer I'm working in

## 3. Technical Verification
- **Container Compatibility**: Code works in Docker environment
- **No Core Changes**: Haven't modified files in `.llm/core/` directory
- **Proper Dependencies**: Using correct service dependencies according to architecture
- **Database Operations**: Using DatabaseStoreAccess for data operations, not direct DB calls

## 4. Cleanup and Testing
- **Temporary Files Removed**: Any scripts or test files created during development are cleaned up
- **Functionality Verified**: The solution actually solves the user's problem
- **Minimal Implementation**: Haven't over-engineered or added unnecessary complexity
- **Error Cases Handled**: Appropriate exception handling for edge cases

## 5. Communication
- **Clear Explanation**: Explained what was changed and why
- **Next Steps Identified**: If applicable, mentioned what the user might want to do next
- **Questions Answered**: Addressed all parts of the user's original request

If any of these checks fail, fix the issues before responding to the user.