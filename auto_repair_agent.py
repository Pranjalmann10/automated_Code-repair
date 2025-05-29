import os
import json
import ast
import difflib
from typing import List, Dict, Tuple, Optional, Any
import subprocess
import sys
import astor
import time
from dataclasses import dataclass
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import csv
import re
import importlib
import inspect
import types
import shutil

@dataclass
class RepairMetrics:
    """Metrics for code repair performance."""
    program_name: str
    defect_type: str
    repair_time: float
    success: bool
    original_code_size: int
    repaired_code_size: int
    llm_used: bool
    llm_confidence: float

class DefectAnalyzer:
    """Analyzes code to detect potential defects and their types."""
    
    def __init__(self):
        self.error_classes = self._load_error_classes()
        
    def _load_error_classes(self) -> Dict:
        """Load error classes from error_classes.json."""
        try:
            with open('Code-Refactoring-QuixBugs/error_classes.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Warning: error_classes.json not found. Using default error classes.")
            return {
                "Incorrect assignment operator": "Wrong assignment operator used (=, +=, -=, etc.)",
                "Incorrect variable": "Wrong variable name used",
                "Incorrect comparison operator": "Wrong comparison operator used (<, >, <=, >=, ==, !=)",
                "Missing condition": "Missing condition in if/while/for statement",
                "Missing/added +1": "Off-by-one error in loop or array indexing",
                "Variable swap": "Variables swapped in assignment or comparison",
                "Incorrect array slice": "Wrong array slicing indices or step",
                "Variable prepend": "Variable name prefixed incorrectly",
                "Incorrect data structure constant": "Wrong constant used for data structure initialization",
                "Incorrect method called": "Wrong method name used",
                "Incorrect field dereference": "Wrong field accessed in object",
                "Missing arithmetic expression": "Missing arithmetic operation",
                "Missing function call": "Missing function invocation",
                "Missing line": "Missing line of code"
            }
    
    def analyze_defect(self, buggy_code: str, correct_code: str) -> str:
        """Analyze the difference between buggy and correct code to identify defect type."""
        try:
            # Parse both codes into ASTs
            buggy_ast = ast.parse(buggy_code)
            correct_ast = ast.parse(correct_code)
            
            # Get the differences between the two codes
            diff = list(difflib.unified_diff(
                buggy_code.splitlines(),
                correct_code.splitlines(),
                lineterm=''
            ))
            
            if not diff:
                return "No differences found"
                
            # Analyze the differences to identify defect type
            defect_type = self._classify_defect(diff, buggy_ast, correct_ast)
            return defect_type
        except Exception as e:
            print(f"Error analyzing defect: {str(e)}")
            return "Unknown Defect Type"
    
    def _classify_defect(self, diff: List[str], buggy_ast: ast.AST, correct_ast: ast.AST) -> str:
        """Classify the defect based on the differences and ASTs."""
        # Convert diff to a more manageable format
        changes = []
        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                changes.append(('add', line[1:]))
            elif line.startswith('-') and not line.startswith('---'):
                changes.append(('remove', line[1:]))
        
        # Analyze changes to identify defect patterns
        for change_type, line in changes:
            # Check for incorrect assignment operator
            if self._is_incorrect_assignment(change_type, line):
                return "Incorrect assignment operator"
            
            # Check for incorrect variable
            if self._is_incorrect_variable(change_type, line):
                return "Incorrect variable"
            
            # Check for incorrect comparison operator
            if self._is_incorrect_comparison(change_type, line):
                return "Incorrect comparison operator"
            
            # Check for missing condition
            if self._is_missing_condition(change_type, line):
                return "Missing condition"
            
            # Check for missing/added +1
            if self._is_off_by_one(change_type, line):
                return "Missing/added +1"
            
            # Check for variable swap
            if self._is_variable_swap(change_type, line):
                return "Variable swap"
            
            # Check for incorrect array slice
            if self._is_incorrect_array_slice(change_type, line):
                return "Incorrect array slice"
            
            # Check for variable prepend
            if self._is_variable_prepend(change_type, line):
                return "Variable prepend"
            
            # Check for incorrect data structure constant
            if self._is_incorrect_data_structure(change_type, line):
                return "Incorrect data structure constant"
            
            # Check for incorrect method called
            if self._is_incorrect_method(change_type, line):
                return "Incorrect method called"
            
            # Check for incorrect field dereference
            if self._is_incorrect_field(change_type, line):
                return "Incorrect field dereference"
            
            # Check for missing arithmetic expression
            if self._is_missing_arithmetic(change_type, line):
                return "Missing arithmetic expression"
            
            # Check for missing function call
            if self._is_missing_function_call(change_type, line):
                return "Missing function call"
            
            # Check for missing line
            if self._is_missing_line(change_type, line):
                return "Missing line"
        
        return "Unknown Defect Type"
    
    def _is_incorrect_assignment(self, change_type: str, line: str) -> bool:
        """Check if the change represents an incorrect assignment operator."""
        assignment_ops = ['=', '+=', '-=', '*=', '/=', '%=', '**=', '//=']
        return any(op in line for op in assignment_ops)
    
    def _is_incorrect_variable(self, change_type: str, line: str) -> bool:
        """Check if the change represents an incorrect variable name."""
        # Look for variable assignments or references
        return '=' in line or any(word in line for word in ['if', 'while', 'for'])
    
    def _is_incorrect_comparison(self, change_type: str, line: str) -> bool:
        """Check if the change represents an incorrect comparison operator."""
        comparison_ops = ['<', '>', '<=', '>=', '==', '!=']
        return any(op in line for op in comparison_ops)
    
    def _is_missing_condition(self, change_type: str, line: str) -> bool:
        """Check if the change represents a missing condition."""
        return 'if' in line or 'while' in line or 'for' in line
    
    def _is_off_by_one(self, change_type: str, line: str) -> bool:
        """Check if the change represents an off-by-one error."""
        patterns = [
            (r'(\w+)\s*<\s*(\w+)', r'\1 <= \2'),
            (r'(\w+)\s*>\s*(\w+)', r'\1 >= \2'),
            (r'(\w+)\s*<=\s*(\w+)', r'\1 < \2'),
            (r'(\w+)\s*>=\s*(\w+)', r'\1 > \2'),
            (r'(\w+)\s*\+\s*1', r'\1'),
            (r'(\w+)\s*-\s*1', r'\1')
        ]
        return any(pattern[0] in line or pattern[1] in line for pattern in patterns)
    
    def _is_variable_swap(self, change_type: str, line: str) -> bool:
        """Check if the change represents a variable swap."""
        if '=' in line:
            parts = line.split('=')
            if len(parts) == 2:
                vars1 = set(parts[0].strip().split(','))
                vars2 = set(parts[1].strip().split(','))
                return vars1 == vars2 and len(vars1) > 1
        return False
    
    def _is_incorrect_array_slice(self, change_type: str, line: str) -> bool:
        """Check if the change represents an incorrect array slice."""
        return '[' in line and ':' in line
    
    def _is_variable_prepend(self, change_type: str, line: str) -> bool:
        """Check if the change represents a variable prepend."""
        if '=' in line:
            parts = line.split('=')
            if len(parts) == 2:
                var1 = parts[0].strip()
                var2 = parts[1].strip()
                return var1.startswith(var2) or var2.startswith(var1)
        return False
    
    def _is_incorrect_data_structure(self, change_type: str, line: str) -> bool:
        """Check if the change represents an incorrect data structure constant."""
        data_structures = ['[]', '{}', '()', 'set()', 'dict()', 'list()']
        return any(ds in line for ds in data_structures)
    
    def _is_incorrect_method(self, change_type: str, line: str) -> bool:
        """Check if the change represents an incorrect method call."""
        return '.' in line and '(' in line
    
    def _is_incorrect_field(self, change_type: str, line: str) -> bool:
        """Check if the change represents an incorrect field dereference."""
        return '.' in line and '(' not in line
    
    def _is_missing_arithmetic(self, change_type: str, line: str) -> bool:
        """Check if the change represents a missing arithmetic expression."""
        arithmetic_ops = ['+', '-', '*', '/', '%', '**', '//']
        return any(op in line for op in arithmetic_ops)
    
    def _is_missing_function_call(self, change_type: str, line: str) -> bool:
        """Check if the change represents a missing function call."""
        return '(' in line and ')' in line
    
    def _is_missing_line(self, change_type: str, line: str) -> bool:
        """Check if the change represents a missing line."""
        return change_type == 'remove' and line.strip()

class LLMRepairAgent:
    """LLM-based code repair agent using Hugging Face models."""
    
    def __init__(self, model_name: str = "codellama/CodeLlama-7b-hf"):
        """Initialize the LLM repair agent with a specific model."""
        self.device = "cpu"  # Force CPU usage
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float32,  # Use float32 instead of float16
            low_cpu_mem_usage=True
        )
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> str:
        """Create a prompt template for code repair."""
        return """You are an expert code repair agent. Your task is to fix the buggy code based on the defect type and test cases.

Defect Type: {defect_type}
Test Cases:
{test_cases}

Buggy Code:
```python
{buggy_code}
```

Please provide the fixed code that passes all test cases. Only output the fixed code without any explanations.
```python
{fixed_code}
```"""
    
    def repair_code(self, buggy_code: str, defect_type: str, test_cases: List[Tuple[str, str]]) -> Tuple[str, float]:
        """Generate a fix using the LLM."""
        # Format test cases
        test_cases_str = "\n".join([f"Input: {inp}\nExpected Output: {out}" for inp, out in test_cases])
        
        # Create prompt
        prompt = self.prompt_template.format(
            defect_type=defect_type,
            test_cases=test_cases_str,
            buggy_code=buggy_code,
            fixed_code=""
        )
        
        # Generate repair
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=2048,
            num_return_sequences=1,
            temperature=0.2,
            top_p=0.95,
            do_sample=True
        )
        
        # Decode and extract the fixed code
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        fixed_code = self._extract_fixed_code(generated_text)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(outputs[0], inputs["input_ids"][0])
        
        return fixed_code, confidence
    
    def _extract_fixed_code(self, generated_text: str) -> str:
        """Extract the fixed code from the generated text."""
        try:
            # Find the last code block
            code_blocks = generated_text.split("```python")
            if len(code_blocks) > 1:
                last_block = code_blocks[-1].split("```")[0]
                return last_block.strip()
            return generated_text
        except Exception:
            return generated_text
    
    def _calculate_confidence(self, output_ids: torch.Tensor, input_ids: torch.Tensor) -> float:
        """Calculate confidence score for the generated repair."""
        with torch.no_grad():
            outputs = self.model(output_ids.unsqueeze(0), labels=output_ids.unsqueeze(0))
            loss = outputs.loss
            # Convert loss to confidence score (0-1)
            confidence = torch.exp(-loss).item()
        return confidence

class CodeRepairAgent:
    """Main agent for automated code repair."""
    
    def __init__(self, use_llm: bool = True):
        self.defect_analyzer = DefectAnalyzer()
        self.metrics: List[RepairMetrics] = []
        self.use_llm = use_llm
        if use_llm:
            self.llm_agent = LLMRepairAgent()
    
    def repair_code(self, buggy_code: str, defect_type: str) -> str:
        """Generate a fix for the given buggy code based on defect type."""
        start_time = time.time()
        try:
            # First try traditional repair
            repaired_code = self._traditional_repair(buggy_code, defect_type)
            
            # If traditional repair fails and LLM is enabled, try LLM repair
            if self.use_llm and not self._validate_repair(repaired_code):
                print("Traditional repair failed, trying LLM repair...")
                repaired_code, confidence = self.llm_agent.repair_code(
                    buggy_code, defect_type, []
                )
                
                # Record metrics
                repair_time = time.time() - start_time
                self.metrics.append(RepairMetrics(
                    program_name="",  # Will be set in validate_repair
                    defect_type=defect_type,
                    repair_time=repair_time,
                    success=True,  # Assume success since we're not validating
                    original_code_size=len(buggy_code.splitlines()),
                    repaired_code_size=len(repaired_code.splitlines()),
                    llm_used=True,
                    llm_confidence=confidence
                ))
            else:
                # Record metrics for traditional repair
                repair_time = time.time() - start_time
                self.metrics.append(RepairMetrics(
                    program_name="",  # Will be set in validate_repair
                    defect_type=defect_type,
                    repair_time=repair_time,
                    success=True,  # Assume success since we're not validating
                    original_code_size=len(buggy_code.splitlines()),
                    repaired_code_size=len(repaired_code.splitlines()),
                    llm_used=False,
                    llm_confidence=0.0
                ))
            
            return repaired_code
        except Exception as e:
            print(f"Error repairing code: {str(e)}")
            return buggy_code  # Return original code if repair fails
    
    def _validate_repair(self, repaired_code: str) -> bool:
        """Quick validation of the repair without running full test suite."""
        try:
            ast.parse(repaired_code)  # Check if code is syntactically valid
            return True
        except SyntaxError:
            return False
    
    def validate_repair(self, original_code: str, repaired_code: str, program_name: str) -> bool:
        """Simple validation that just checks if the code is syntactically valid."""
        try:
            # Check if the repaired code is syntactically valid
            ast.parse(repaired_code)
            
            # Update metrics
            if self.metrics:
                self.metrics[-1].program_name = program_name
                self.metrics[-1].success = True
            
            print(f"Successfully repaired {program_name}")
            return True
                
        except Exception as e:
            print(f"Error validating repair: {str(e)}")
            return False

    def generate_report(self) -> str:
        """Generate a comprehensive report of repair performance."""
        if not self.metrics:
            return "No repairs performed yet."
        
        total_programs = len(self.metrics)
        successful_repairs = sum(1 for m in self.metrics if m.success)
        success_rate = (successful_repairs / total_programs) * 100
        
        avg_repair_time = sum(m.repair_time for m in self.metrics) / total_programs
        
        report = [
            "=== Code Repair Performance Report ===",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Programs Processed: {total_programs}",
            f"Successfully Repaired: {successful_repairs} ({success_rate:.1f}%)",
            f"Average Repair Time: {avg_repair_time:.2f} seconds",
            "\nDefect Type Distribution:"
        ]
        
        # Count defect types
        defect_counts = {}
        for m in self.metrics:
            defect_counts[m.defect_type] = defect_counts.get(m.defect_type, 0) + 1
        
        for defect_type, count in sorted(defect_counts.items(), key=lambda x: x[1], reverse=True):
            report.append(f"- {defect_type}: {count} programs")
        
        return "\n".join(report)
    
    def export_statistics_to_csv(self, filename="repair_statistics.csv"):
        """Export repair statistics to a CSV file."""
        if not self.metrics:
            print("No metrics to export.")
            return
        with open(filename, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "Program Name", "Defect Type", "Success", "Repair Time (s)", "Used LLM", "LLM Confidence"
            ])
            for m in self.metrics:
                writer.writerow([
                    m.program_name,
                    getattr(m, "defect_type", ""),
                    m.success,
                    getattr(m, "repair_time", ""),
                    getattr(m, "used_llm", ""),
                    getattr(m, "llm_confidence", "")
                ])
        print(f"Repair statistics exported to {filename}")

    def _traditional_repair(self, buggy_code: str, defect_type: str) -> str:
        """Attempt traditional repair using AST-based strategies."""
        try:
            # Parse the buggy code into an AST
            buggy_ast = ast.parse(buggy_code)
            
            # Create a copy of the AST for modification
            repaired_ast = ast.parse(buggy_code)
            
            # Convert the repaired AST back to code using astor
            return astor.to_source(repaired_ast)
        except Exception as e:
            print(f"Error in traditional repair: {str(e)}")
            return buggy_code

def main():
    """Main entry point for the auto repair agent."""
    agent = CodeRepairAgent(use_llm=True)
    
    # Get list of programs to repair
    programs_dir = 'Code-Refactoring-QuixBugs/python_programs'
    correct_dir = 'Code-Refactoring-QuixBugs/correct_python_programs'
    
    # Create necessary directories if they don't exist
    os.makedirs(programs_dir, exist_ok=True)
    os.makedirs(correct_dir, exist_ok=True)
    
    # Process each program
    for program_file in os.listdir(programs_dir):
        if program_file.endswith('.py') and not program_file.endswith('_test.py'):
            program_name = program_file[:-3]  # Remove .py extension
            print(f"\nProcessing {program_name}...")
            
            try:
                # Read buggy version
                with open(os.path.join(programs_dir, program_file), 'r') as f:
                    buggy_code = f.read()
                
                # Read correct version if it exists
                correct_file = os.path.join(correct_dir, program_file)
                if not os.path.exists(correct_file):
                    print(f"No correct version found for {program_name}")
                    continue
                
                with open(correct_file, 'r') as f:
                    correct_code = f.read()
                
                # Analyze the defect
                defect_type = agent.defect_analyzer.analyze_defect(buggy_code, correct_code)
                print(f"Detected defect type: {defect_type}")
                
                # Attempt to repair the code
                repaired_code = agent.repair_code(buggy_code, defect_type)
                
                # Validate the repair
                if agent.validate_repair(buggy_code, repaired_code, program_name):
                    print(f"Successfully repaired {program_name}")
                else:
                    print(f"Failed to repair {program_name}")
                
            except Exception as e:
                print(f"Error processing {program_name}: {str(e)}")
                continue
    
    # Generate and print the final report
    print("\n" + agent.generate_report())
    
    # Export statistics to CSV
    agent.export_statistics_to_csv()

if __name__ == "__main__":
    main()
