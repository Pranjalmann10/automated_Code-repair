# Automated Code Repair Agent

This project implements an automated code repair agent that can detect and fix common programming defects in Python code. The agent uses a combination of traditional AST-based analysis and Large Language Models (LLMs) to perform code repairs.

## Features

- **Defect Analysis**: Detects common programming defects such as:
  - Incorrect assignment operators
  - Incorrect variables
  - Missing function calls
  - Incorrect method calls
  - Missing lines
  - Incorrect data structure constants
  - Incorrect array slices
  - Missing arithmetic expressions

- **Code Repair**: Uses two approaches for code repair:
  1. Traditional AST-based repair
  2. LLM-based repair using CodeLlama-7b-hf model

- **Metrics Tracking**: Tracks repair performance metrics including:
  - Repair success rate
  - Repair time
  - Defect type distribution
  - LLM usage statistics

## Project Structure

```
.
├── auto_repair_agent.py      # Main code repair agent implementation
├── requirements.txt          # Project dependencies
└── Code-Refactoring-QuixBugs/
    ├── python_programs/      # Buggy Python programs
    ├── correct_python_programs/  # Correct versions of programs
    ├── fixed_programs/       # Repaired program versions
    └── error_classes.json    # Defect type definitions
```

## Dependencies

- Python 3.8+
- transformers
- torch
- astor
- networkx
- numpy

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

1. Place buggy Python programs in the `Code-Refactoring-QuixBugs/python_programs/` directory
2. Place corresponding correct versions in `Code-Refactoring-QuixBugs/correct_python_programs/`
3. Run the repair agent:
```bash
python auto_repair_agent.py
```

The agent will:
1. Process each program
2. Detect defect types
3. Attempt repairs using both traditional and LLM-based approaches
4. Generate a performance report
5. Export statistics to `repair_statistics.csv`

## Implementation Details

### CodeRepairAgent

The main agent class that orchestrates the repair process:
- Analyzes defects using AST-based analysis
- Attempts repairs using traditional methods first
- Falls back to LLM-based repair if traditional repair fails
- Tracks repair metrics and generates reports

### DefectAnalyzer

Analyzes code to detect common programming defects:
- Uses AST analysis to identify defect patterns
- Compares buggy code with correct versions
- Classifies defects into predefined categories

### LLMRepairAgent

Handles LLM-based code repairs:
- Uses CodeLlama-7b-hf model for code generation
- Provides repair suggestions based on defect type
- Includes confidence scores for repairs

## Performance Metrics

The agent tracks various metrics:
- Total programs processed
- Success rate
- Average repair time
- Defect type distribution
- LLM usage statistics

## Limitations

- Currently focuses on syntax validation rather than functional correctness
- LLM-based repairs may be computationally intensive
- Some complex defects may require manual intervention

## Future Improvements

- Add more sophisticated defect detection
- Implement test case validation
- Support for more programming languages
- Enhanced LLM integration
- Better error handling and recovery

## License

This project is licensed under the MIT License - see the LICENSE file for details. 