"""
Output Formatter Module
Formats RAG responses into structured JSON based on subject type.
"""
import json
import re
from typing import Dict, Any, List, Optional


class OutputFormatter:
    """Formats responses into structured JSON based on subject."""
    
    def __init__(self):
        pass
    
    def format_general_response(
        self,
        response_text: str,
        query: str,
        subject: str
    ) -> Dict[str, Any]:
        """
        Format a general response into structured JSON.
        
        Args:
            response_text: Raw response from LLM
            query: Original query
            subject: Subject category
            
        Returns:
            Structured JSON response
        """
        # Try to parse as JSON first
        try:
            parsed = json.loads(response_text)
            return self._validate_general_schema(parsed)
        except json.JSONDecodeError:
            pass
        
        # If not valid JSON, try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return self._validate_general_schema(parsed)
            except json.JSONDecodeError:
                pass
        
        # Fallback: Create structured response from plain text
        return self._create_general_from_text(response_text, query)
    
    def format_math_response(
        self,
        response_text: str,
        query: str
    ) -> Dict[str, Any]:
        """
        Format a math response into step-by-step JSON.
        
        Args:
            response_text: Raw response from LLM
            query: Original math problem
            
        Returns:
            Structured JSON with steps
        """
        # Try to parse as JSON first
        try:
            parsed = json.loads(response_text)
            return self._validate_math_schema(parsed)
        except json.JSONDecodeError:
            pass
        
        # If not valid JSON, try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return self._validate_math_schema(parsed)
            except json.JSONDecodeError:
                pass
        
        # Fallback: Create structured response from plain text
        return self._create_math_from_text(response_text, query)
    
    def _validate_general_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize general response schema."""
        result = {
            "summary": data.get("summary", ""),
            "caption": data.get("caption", "Response"),
            "bullet_points": [],
            "table": []
        }
        
        # Process bullet points
        if "bullet_points" in data:
            for point in data["bullet_points"]:
                if isinstance(point, dict):
                    result["bullet_points"].append(point)
                elif isinstance(point, str):
                    result["bullet_points"].append({"point": point})
        
        # Process table
        if "table" in data and isinstance(data["table"], list):
            result["table"] = data["table"]
        
        return result
    
    def _validate_math_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize math response schema."""
        result = {
            "problem": data.get("problem", ""),
            "caption": data.get("caption", "Math Solution"),
            "steps": [],
            "final_answer": data.get("final_answer", ""),
            "concept_used": data.get("concept_used", []),
            "tips": data.get("tips", [])
        }
        
        # Process steps
        if "steps" in data:
            for i, step in enumerate(data["steps"], 1):
                if isinstance(step, dict):
                    validated_step = {
                        "step_number": step.get("step_number", i),
                        "action": step.get("action", ""),
                        "explanation": step.get("explanation", ""),
                        "expression": step.get("expression", ""),
                        "result": step.get("result", "")
                    }
                    result["steps"].append(validated_step)
        
        return result
    
    def _create_general_from_text(self, text: str, query: str) -> Dict[str, Any]:
        """Create general response structure from plain text."""
        # Split text into sentences for bullet points
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # First sentence as summary, rest as bullet points
        summary = sentences[0] if sentences else text
        bullet_points = [{"point": s} for s in sentences[1:5]]  # Max 5 points
        
        # Generate caption from query
        caption = self._generate_caption(query)
        
        return {
            "summary": summary,
            "caption": caption,
            "bullet_points": bullet_points,
            "table": []
        }
    
    def _create_math_from_text(self, text: str, query: str) -> Dict[str, Any]:
        """Create math response structure from plain text."""
        # Try to extract steps from numbered patterns
        step_patterns = [
            r'(?:Step\s*)?(\d+)[.:\)]\s*(.+?)(?=(?:Step\s*)?\d+[.:\)]|$)',
            r'(\d+)\.\s*(.+?)(?=\d+\.|$)',
        ]
        
        steps = []
        for pattern in step_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                for i, match in enumerate(matches, 1):
                    step_text = match[1].strip() if len(match) > 1 else match[0].strip()
                    steps.append({
                        "step_number": i,
                        "action": f"Step {i}",
                        "explanation": step_text,
                        "expression": "",
                        "result": ""
                    })
                break
        
        # If no steps found, create a single step
        if not steps:
            steps = [{
                "step_number": 1,
                "action": "Solution",
                "explanation": text[:500],
                "expression": "",
                "result": ""
            }]
        
        # Try to extract final answer
        answer_match = re.search(r'(?:answer|result|solution)[:\s]*(.+?)(?:[.\n]|$)', text, re.IGNORECASE)
        final_answer = answer_match.group(1).strip() if answer_match else ""
        
        return {
            "problem": query,
            "caption": self._generate_caption(query),
            "steps": steps,
            "final_answer": final_answer,
            "concept_used": [],
            "tips": []
        }
    
    def _generate_caption(self, query: str) -> str:
        """Generate a caption from the query."""
        # Remove question words and clean up
        caption = re.sub(r'^(what|how|why|explain|describe|solve|find|calculate)\s+', '', query, flags=re.IGNORECASE)
        caption = re.sub(r'\?$', '', caption)
        caption = caption.strip().title()
        
        if len(caption) > 50:
            caption = caption[:50] + "..."
        
        return caption if caption else "Response"
    
    def to_json_string(self, response: Dict[str, Any], indent: int = 2) -> str:
        """Convert response to formatted JSON string."""
        return json.dumps(response, indent=indent, ensure_ascii=False)


if __name__ == "__main__":
    # Test the output formatter
    formatter = OutputFormatter()
    
    # Test general response
    general_text = """
    {
        "summary": "Alcohol has several important properties including high boiling point and good conductivity.",
        "caption": "Properties of Alcohol",
        "bullet_points": [
            {"point": "High boiling point (357°C)"},
            {"point": "Low freezing point (-39°C)"}
        ],
        "table": []
    }
    """
    
    result = formatter.format_general_response(general_text, "What are properties of alcohol?", "Science")
    print("General response:")
    print(formatter.to_json_string(result))
    
    # Test math response
    math_text = """
    {
        "problem": "Solve: 2x + 5 = 15",
        "caption": "Solving Linear Equation",
        "steps": [
            {
                "step_number": 1,
                "action": "Subtract 5 from both sides",
                "explanation": "To isolate the variable term",
                "expression": "2x = 10",
                "result": "2x = 10"
            }
        ],
        "final_answer": "x = 5",
        "concept_used": ["Linear equations"],
        "tips": ["Always perform same operation on both sides"]
    }
    """
    
    result = formatter.format_math_response(math_text, "Solve: 2x + 5 = 15")
    print("\nMath response:")
    print(formatter.to_json_string(result))
