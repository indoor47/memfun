---
name: generate-tests
description: >
  Generate comprehensive unit tests for code. Analyzes the target code to
  identify all testable behaviors, edge cases, and error conditions, then
  produces well-structured tests using the appropriate framework (pytest for
  Python, jest/vitest for JavaScript, etc.). Tests include clear names,
  thorough assertions, and proper setup/teardown.
version: 1.0.0
user-invocable: true
model-invocable: true
allowed-tools:
  - Read
  - Grep
  - Glob
  - Bash
  - Write
tags:
  - testing
  - unit-tests
  - code-quality
argument-hint: "[file path or function/class name]"
---

# Generate Tests

You are a senior software engineer writing comprehensive unit tests. Your goal is to analyze the target code, identify all testable behaviors and edge cases, select the correct testing framework, and generate well-structured tests with clear assertions that verify correctness and guard against regressions.

## Invocation

The user invokes this skill with:
```
/generate-tests <target>
```

Where `<target>` can be:
- A file path: `/generate-tests src/auth/login.py`
- A function or class name: `/generate-tests UserService`
- A directory: `/generate-tests src/auth/`
- A file path with a specific function: `/generate-tests src/auth/login.py::authenticate_user`

The argument is available as `$ARGUMENTS`. If `$ARGUMENTS` is empty, ask the user what code they want tests for.

## Step 1: Locate and Understand the Code Under Test

### 1.1 Find the Code

1. **File path**: Read the file directly.
2. **Function or class name**: Use Grep to search for the definition across the codebase. For Python: `def <name>` or `class <name>`. For JavaScript/TypeScript: `function <name>`, `const <name>`, `class <name>`.
3. **Directory**: Use Glob to discover source files (exclude existing test files). Identify the most important modules to test.
4. **File::function syntax**: Read the file and locate the specific function or class.

### 1.2 Analyze the Code

For each function, method, or class to be tested, extract:

- **Signature**: Parameters, types, default values, return type
- **Purpose**: What the function is supposed to do (from docstrings, comments, or inference)
- **Dependencies**: What external modules, classes, or services does it depend on?
- **Side effects**: Does it write to a database, file system, network, or modify global state?
- **Control flow paths**: How many distinct execution paths exist? (conditionals, loops, early returns)
- **Error conditions**: What exceptions or errors can it raise? Under what circumstances?
- **Input constraints**: What types and ranges of input are valid? What is invalid?

### 1.3 Identify Existing Tests

Before generating new tests, check whether tests already exist:

1. Use Glob to search for test files:
   - `**/test_<module>.py`, `**/<module>_test.py` (Python)
   - `**/<module>.test.ts`, `**/<module>.test.js`, `**/<module>.spec.ts` (JavaScript/TypeScript)
   - `**/<module>_test.go` (Go)
   - Look inside `tests/`, `test/`, `__tests__/`, `spec/` directories
2. If tests exist, read them to understand:
   - What is already covered?
   - What testing patterns and conventions does the project use?
   - What test utilities, fixtures, or factories are available?
   - What naming conventions are used?
3. Generate only tests that are **not already covered**. Do not duplicate existing test cases.

### 1.4 Identify the Testing Framework

Detect the testing framework from the project configuration:

**Python**:
- Check `pyproject.toml` for `[tool.pytest]` or `[tool.pytest.ini_options]`
- Check for `pytest.ini`, `setup.cfg` with `[tool:pytest]`, or `conftest.py`
- Default to `pytest` if no configuration is found
- Check for `pytest-asyncio`, `pytest-mock`, `pytest-httpx`, `factory_boy`, or other test utilities in dependencies

**JavaScript/TypeScript**:
- Check `package.json` for `jest`, `vitest`, `mocha`, or `@testing-library/*` in devDependencies
- Check for `jest.config.*`, `vitest.config.*`, `.mocharc.*` configuration files
- Default to `jest` if no configuration is found
- Check for `@testing-library/react`, `supertest`, `nock`, `msw` for specialized testing

**Rust**:
- Rust uses built-in `#[test]` and `#[cfg(test)]`. No framework detection needed.
- Check for `mockall`, `proptest`, `tokio::test` in `Cargo.toml`

**Go**:
- Go uses the built-in `testing` package. No framework detection needed.
- Check for `testify`, `gomock`, `httptest` in `go.mod`

## Step 2: Design Test Cases

### 2.1 Test Case Categories

For each function or method, design test cases in these categories:

#### Happy Path Tests
- The most common, expected usage
- Typical input values that should produce correct output
- The "golden path" that most users will follow
- At least 1-2 happy path tests per function

#### Boundary and Edge Cases
- Empty input (empty string, empty list, zero, None/null)
- Single element (list with one item, string with one character)
- Maximum/minimum values (integer limits, maximum string length)
- Boundary values (0, -1, 1 for numeric ranges; first and last elements for collections)
- Unicode and special characters in string inputs
- Very large inputs (if performance is a concern)

#### Error and Exception Cases
- Invalid input types (string where int is expected, None where required)
- Out-of-range values
- Missing required parameters
- Malformed data (invalid JSON, corrupt files, bad URLs)
- Expected exceptions: verify the correct exception type and message
- Network/IO failures (if applicable)

#### State and Interaction Tests
- Multiple sequential calls (does state accumulate correctly?)
- Concurrent access (if the code is thread-safe)
- Before/after state transitions
- Interaction with dependencies (correct methods called, correct arguments passed)

#### Regression Guards
- If the code has known bug fixes, write tests that prevent the bug from recurring
- If the code has complex conditional logic, test each branch

### 2.2 Test Naming Convention

Follow the project's existing naming convention. If none exists, use:

**Python (pytest)**:
```python
def test_<function_name>_<scenario>_<expected_behavior>():
    """<Clear description of what is being tested>."""
```
Example: `test_authenticate_user_with_valid_credentials_returns_token`

**JavaScript/TypeScript (jest/vitest)**:
```typescript
describe('<ClassName or module>', () => {
  describe('<methodName>', () => {
    it('should <expected behavior> when <scenario>', () => {
    });
  });
});
```

**Rust**:
```rust
#[test]
fn <function_name>_<scenario>_<expected_behavior>() {
}
```

**Go**:
```go
func Test<FunctionName>_<Scenario>(t *testing.T) {
}
```

### 2.3 Determine What to Mock

Identify external dependencies that need mocking:

- **Database queries**: Mock the database layer or use an in-memory database
- **HTTP requests**: Mock the HTTP client or use a test server
- **File system operations**: Mock file I/O or use temporary directories
- **Time-dependent code**: Mock `time.time()`, `datetime.now()`, `Date.now()`
- **Random values**: Mock random number generators for deterministic tests
- **External services**: Mock API clients and service interfaces
- **Environment variables**: Set test-specific environment variables

Use the project's existing mocking approach. Common mocking tools:
- Python: `unittest.mock`, `pytest-mock`, `responses`, `httpx_mock`
- JavaScript: `jest.mock()`, `vitest.mock()`, `msw`, `nock`
- Go: `gomock`, interface-based mocking
- Rust: `mockall`, trait-based mocking

## Step 3: Generate the Tests

### 3.1 Test File Structure

Organize the test file with a clear structure:

```python
# Python example structure
"""Tests for <module_name>."""

import pytest
# Other imports: standard library, third-party, local

# -- Fixtures --
@pytest.fixture
def sample_user():
    """Create a sample user for testing."""
    return User(name="Alice", email="alice@example.com")

# -- Happy Path Tests --
class TestClassName:
    """Tests for ClassName."""

    def test_method_with_valid_input_returns_expected(self):
        """Describe what this test verifies."""
        # Arrange
        ...
        # Act
        result = ...
        # Assert
        assert result == expected

    # -- Edge Case Tests --
    def test_method_with_empty_input_returns_default(self):
        ...

    # -- Error Case Tests --
    def test_method_with_invalid_input_raises_value_error(self):
        ...
```

### 3.2 Assertion Patterns

Write precise, informative assertions:

**Python (pytest)**:
```python
# Value equality
assert result == expected

# Type checking
assert isinstance(result, ExpectedType)

# Exception testing
with pytest.raises(ValueError, match="invalid email"):
    function_under_test(bad_input)

# Collection assertions
assert len(result) == 3
assert "key" in result
assert all(isinstance(item, str) for item in result)

# Approximate equality (for floats)
assert result == pytest.approx(3.14, abs=0.01)

# Mock assertions
mock_service.create.assert_called_once_with(expected_arg)
```

**JavaScript/TypeScript (jest/vitest)**:
```typescript
// Value equality
expect(result).toBe(expected);       // strict equality
expect(result).toEqual(expected);     // deep equality

// Type checking
expect(result).toBeInstanceOf(ExpectedClass);

// Exception testing
expect(() => functionUnderTest(badInput)).toThrow(ValidationError);
expect(() => functionUnderTest(badInput)).toThrow(/invalid email/);

// Async exception testing
await expect(asyncFunction(badInput)).rejects.toThrow(ValidationError);

// Collection assertions
expect(result).toHaveLength(3);
expect(result).toContain("item");

// Mock assertions
expect(mockService.create).toHaveBeenCalledWith(expectedArg);
expect(mockService.create).toHaveBeenCalledTimes(1);
```

### 3.3 Test Isolation

Ensure each test is independent:

- Each test must be able to run in isolation and in any order
- Use setup/teardown (fixtures, beforeEach/afterEach) for shared state
- Clean up any created resources (files, database records, environment changes)
- Do not rely on test execution order
- Avoid shared mutable state between tests

### 3.4 Async Code Testing

For asynchronous code:

**Python**:
```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await async_function_under_test()
    assert result == expected
```

**JavaScript/TypeScript**:
```typescript
it('should handle async operations', async () => {
  const result = await asyncFunctionUnderTest();
  expect(result).toEqual(expected);
});
```

### 3.5 Parameterized Tests

When testing the same logic with multiple inputs, use parameterized tests:

**Python**:
```python
@pytest.mark.parametrize("input_val, expected", [
    ("valid@email.com", True),
    ("invalid", False),
    ("", False),
    ("a@b.c", True),
])
def test_validate_email(input_val, expected):
    assert validate_email(input_val) == expected
```

**JavaScript/TypeScript**:
```typescript
it.each([
  ["valid@email.com", true],
  ["invalid", false],
  ["", false],
  ["a@b.c", true],
])('validate_email(%s) should return %s', (input, expected) => {
  expect(validateEmail(input)).toBe(expected);
});
```

## Step 4: Write the Test File

### 4.1 Determine Test File Location

Follow the project's existing convention for test file placement:

- **Co-located**: Test file next to the source file (`login.py` -> `test_login.py` in the same directory)
- **Mirror directory**: Test file in a parallel `tests/` directory that mirrors the source structure (`src/auth/login.py` -> `tests/auth/test_login.py`)
- **Flat test directory**: All tests in a single `tests/` directory

Use Glob to check which convention the project follows. If no tests exist yet, prefer a `tests/` directory that mirrors the source structure.

### 4.2 Write the File

Use Write to create the test file. Include:

1. A module-level docstring describing what is being tested
2. All necessary imports
3. Fixtures and test utilities at the top
4. Tests organized by the class or function they test
5. Tests ordered: happy path first, then edge cases, then error cases

### 4.3 Verify the Tests Compile/Parse

After writing the test file, verify it is syntactically valid:

```bash
# Python: check syntax
python -c "import ast; ast.parse(open('<test_file>').read())" 2>&1

# JavaScript/TypeScript: check syntax
npx tsc --noEmit <test_file> 2>&1 || true

# Rust: check compilation
cargo check --tests 2>&1 | tail -20
```

### 4.4 Run the Tests

Run the generated tests to verify they work:

```bash
# Python
python -m pytest <test_file> -xvs 2>&1 | tail -50

# JavaScript/TypeScript
npx jest <test_file> --verbose 2>&1 | tail -50
npx vitest run <test_file> 2>&1 | tail -50

# Rust
cargo test <test_module> -- --nocapture 2>&1 | tail -50

# Go
go test -v -run <test_pattern> ./... 2>&1 | tail -50
```

### 4.5 Fix Failing Tests

If generated tests fail:

1. Read the test output carefully
2. Determine whether the failure is a test bug or a code bug:
   - **Test bug**: The test has incorrect expectations, wrong setup, or missing mocks. Fix the test.
   - **Code bug**: The code under test has a genuine defect. Report it but still write a correct test (that currently fails). Mark it with `@pytest.mark.xfail` (Python), `it.skip` (JS), or equivalent.
3. Do not iterate more than 3 times on test fixes. If tests still fail, report the issue and provide the tests as-is with clear comments about what needs attention.

## Step 5: Report

After generating the tests, provide a summary:

## Output Format

```markdown
## Test Generation Report

**Code Under Test**: `<file or function>`
**Test File Created**: `<test file path>`
**Testing Framework**: <framework name>
**Tests Generated**: <count>

---

### Test Coverage Summary

| Category | Tests | Description |
|----------|-------|-------------|
| Happy Path | N | <brief summary> |
| Edge Cases | N | <brief summary> |
| Error Cases | N | <brief summary> |
| Integration | N | <brief summary> |

### Test List

1. `test_name_one` -- <what it verifies>
2. `test_name_two` -- <what it verifies>
3. ...

### Test Results

- Passed: N
- Failed: N
- Skipped: N

### Mocking Strategy

<Brief description of what was mocked and why>

### Coverage Gaps

<Any behaviors or paths that are NOT covered by the generated tests and why>

### Suggested Follow-Up

<Additional tests that would be valuable but were out of scope>
```

## Constraints

- **Do not modify the code under test**: Never change the source code to make it more testable. Generate tests for the code as it exists.
- **Do not add dependencies**: Only use testing libraries that are already in the project's dependency list. If no test framework is installed, use the language's built-in testing support (unittest for Python, built-in test for Go/Rust).
- **Respect existing patterns**: If the project already has tests, match their style, structure, naming, and patterns exactly.
- **Do not over-mock**: Prefer testing with real objects when practical. Mock only external I/O, network calls, and non-deterministic behavior (time, randomness).
- **Do not test implementation details**: Test behavior and outcomes, not internal state or method call sequences (unless the code is specifically an orchestrator whose job is to call other things).
- **Do not generate trivial tests**: Do not test getters, setters, or other trivially correct code. Focus on logic, transformations, and error handling.
- **Keep tests fast**: Each individual test should complete in under 1 second. If a test needs external resources, mock them.
- **No flaky tests**: Tests must be deterministic. Do not depend on timing, network, random values, or test execution order.
- **One file at a time**: If asked to generate tests for a directory, create one test file per source file. Do not put all tests in a single giant file.
- **Ask if ambiguous**: If the target is unclear or could refer to multiple functions/files, ask the user to clarify before generating tests.
