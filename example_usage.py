#!/usr/bin/env python3
"""
Example Usage Script: LLM Token Probability Analysis

This script demonstrates the complete workflow of the LLM token probability analysis system.
It shows how to:
1. Generate text with probability analysis
2. Visualize token confidence levels
3. Identify potential problem areas
4. Analyze the relationship between model confidence and code quality

Run this script to see the system in action!
"""

import os
import sys
import argparse
from typing import Optional
import webbrowser
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from LLM import QwenProbabilityAnalyzer, TokenAnalysis
from visualizer import TokenVisualizer, TokenVisualizationMode
from use_case import CodeGenerationUseCase


def simple_text_generation_example(model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"):
    """
    Simple example: Generate text and show basic probability analysis.
    """
    print("="*60)
    print("SIMPLE TEXT GENERATION EXAMPLE")
    print("="*60)
    
    # Initialize analyzer
    print(f"Loading {model_name} model...")
    analyzer = QwenProbabilityAnalyzer(model_name=model_name)
    
    # Simple prompt
    prompt = "Explain how recursion works in programming with a simple example."
    
    print(f"\nPrompt: {prompt}")
    print("\nGenerating text with probability analysis...")
    
    # Generate with analysis
    generated_text, token_analyses = analyzer.generate_with_analysis(
        prompt=prompt,
        max_new_tokens=100,
        temperature=0.7
    )
    
    print("\nGenerated Text:")
    print("-" * 40)
    print(generated_text)
    
    # Show basic statistics
    stats = analyzer.get_generation_stats()
    print("\nGeneration Statistics:")
    print("-" * 40)
    for key, value in stats.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Create visualization
    visualizer = TokenVisualizer()
    
    # Create HTML visualization
    html_viz = visualizer.create_html_visualization(
        token_analyses, 
        mode=TokenVisualizationMode.PROBABILITY,
        title="Simple Text Generation - Probability Analysis"
    )
    
    # Save visualization
    with open("simple_example_visualization.html", "w", encoding="utf-8") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Simple Example - Token Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .stats {{ background-color: #f4f4f4; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Simple Text Generation Example</h1>
            <div class="stats">
                <h3>Statistics:</h3>
                <ul>
                    {"".join([f"<li>{k}: {v:.4f if isinstance(v, float) else v}</li>" for k, v in stats.items()])}
                </ul>
            </div>
            {html_viz}
        </body>
        </html>
        """)
    
    print("\nVisualization saved to 'simple_example_visualization.html'")
    return "simple_example_visualization.html"


def advanced_code_analysis_example(model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"):
    """
    Advanced example: Full code generation analysis with the use case.
    """
    print("="*60)
    print("ADVANCED CODE ANALYSIS EXAMPLE")
    print("="*60)
    
    # Run the complete use case
    use_case = CodeGenerationUseCase()
    results = use_case.run_complete_analysis()
    
    # Save the results
    use_case.save_complete_analysis(results, "advanced_code_analysis.html")
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Model: Qwen 2.5 Coder 7B Instruct")
    print(f"Task: Binary Search Implementation")
    print(f"Generated Tokens: {results['generation_stats']['total_tokens']}")
    print(f"Average Probability: {results['generation_stats']['avg_probability']:.4f}")
    print(f"Minimum Probability: {results['generation_stats']['min_probability']:.4f}")
    
    if results['test_results']['success']:
        print(f"Test Pass Rate: {results['test_results']['pass_rate']:.1%}")
        print(f"Tests Passed: {results['test_results']['passed_tests']}/{results['test_results']['total_tests']}")
    else:
        print(f"Testing Failed: {results['test_results']['error']}")
    
    print(f"Low Confidence Regions: {results['correlation_analysis']['low_confidence_regions_count']}")
    print(f"Correlation Hypothesis: {results['correlation_analysis']['correlation_hypothesis']}")
    
    print("\nKey Insights:")
    for insight in results['correlation_analysis']['insights']:
        print(f"  • {insight}")
    
    print("\nDetailed analysis saved to 'advanced_code_analysis.html'")
    return "advanced_code_analysis.html"


def comparison_example(model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"):
    """
    Example comparing different temperature settings and their effects on confidence.
    """
    print("="*60)
    print("TEMPERATURE COMPARISON EXAMPLE")
    print("="*60)
    
    analyzer = QwenProbabilityAnalyzer(model_name=model_name)
    visualizer = TokenVisualizer()
    
    prompt = "Write a function to check if a number is prime."
    temperatures = [0.1, 0.5, 1.0]
    
    results = {}
    
    for temp in temperatures:
        print(f"\nGenerating with temperature = {temp}...")
        
        generated_text, token_analyses = analyzer.generate_with_analysis(
            prompt=prompt,
            max_new_tokens=80,
            temperature=temp,
            do_sample=True
        )
        
        stats = analyzer.get_generation_stats()
        results[temp] = {
            'text': generated_text,
            'analyses': token_analyses,
            'stats': stats
        }
    
    # Create comparison visualization
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Temperature Comparison Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .comparison { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .temp-result { border: 1px solid #ddd; padding: 20px; border-radius: 8px; }
            .stats { background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>Temperature Comparison: Effect on Model Confidence</h1>
        <p><strong>Prompt:</strong> """ + prompt + """</p>
        <div class="comparison">
    """
    
    for temp in temperatures:
        result = results[temp]
        viz = visualizer.create_html_visualization(
            result['analyses'], 
            mode=TokenVisualizationMode.PROBABILITY,
            title=f"Temperature {temp}"
        )
        
        html_content += f"""
        <div class="temp-result">
            <h3>Temperature = {temp}</h3>
            <div class="stats">
                <strong>Statistics:</strong><br>
                Avg Probability: {result['stats']['avg_probability']:.4f}<br>
                Min Probability: {result['stats']['min_probability']:.4f}<br>
                Max Probability: {result['stats']['max_probability']:.4f}<br>
                Avg Rank: {result['stats']['avg_rank']:.1f}
            </div>
            {viz}
        </div>
        """
    
    html_content += """
        </div>
        <div style="margin-top: 40px;">
            <h2>Analysis</h2>
            <ul>
                <li><strong>Lower Temperature (0.1):</strong> More deterministic, higher confidence, but potentially less creative</li>
                <li><strong>Medium Temperature (0.5):</strong> Balanced between confidence and creativity</li>
                <li><strong>Higher Temperature (1.0):</strong> More creative and diverse, but potentially lower confidence</li>
            </ul>
            <p>This demonstrates how generation parameters affect model confidence patterns.</p>
        </div>
    </body>
    </html>
    """
    
    with open("temperature_comparison.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("\nTemperature comparison saved to 'temperature_comparison.html'")
    
    # Print summary
    print("\nComparison Summary:")
    for temp in temperatures:
        stats = results[temp]['stats']
        print(f"Temperature {temp}: Avg Prob = {stats['avg_probability']:.4f}, "
              f"Avg Rank = {stats['avg_rank']:.1f}")
    
    return "temperature_comparison.html"


def interactive_demo(model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"):
    """
    Interactive demo where user can input their own prompts.
    """
    print("="*60)
    print("INTERACTIVE DEMO")
    print("="*60)
    print("Enter your own prompts to see how the model's confidence varies!")
    print("Type 'quit' to exit the demo.")
    
    analyzer = QwenProbabilityAnalyzer(model_name=model_name)
    visualizer = TokenVisualizer()
    
    demo_count = 0
    
    while True:
        prompt = input("\nEnter your prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Thanks for using the interactive demo!")
            break
        
        if not prompt:
            print("Please enter a valid prompt.")
            continue
        
        print(f"\nGenerating response for: {prompt}")
        
        try:
            generated_text, token_analyses = analyzer.generate_with_analysis(
                prompt=prompt,
                max_new_tokens=100,
                temperature=0.7
            )
            
            print("\nGenerated Text:")
            print("-" * 40)
            print(generated_text)
            
            stats = analyzer.get_generation_stats()
            print(f"\nConfidence Score: {stats['avg_probability']:.4f}")
            print(f"Average Rank: {stats['avg_rank']:.1f}")
            
            # Create visualization
            viz = visualizer.create_html_visualization(
                token_analyses,
                mode=TokenVisualizationMode.PROBABILITY,
                title=f"Interactive Demo - {prompt[:50]}..."
            )
            
            demo_count += 1
            filename = f"interactive_demo_{demo_count}.html"
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Interactive Demo {demo_count}</title>
                    <style>body {{ font-family: Arial, sans-serif; margin: 40px; }}</style>
                </head>
                <body>
                    <h1>Interactive Demo #{demo_count}</h1>
                    <p><strong>Prompt:</strong> {prompt}</p>
                    <p><strong>Confidence Score:</strong> {stats['avg_probability']:.4f}</p>
                    {viz}
                </body>
                </html>
                """)
            
            print(f"Visualization saved to '{filename}'")
            
        except Exception as e:
            print(f"Error during generation: {e}")
            print("Please try a different prompt.")


def main():
    """
    Main function that runs different examples based on command line arguments.
    """
    parser = argparse.ArgumentParser(description="LLM Token Probability Analysis Examples")
    parser.add_argument("--example", choices=["simple", "advanced", "comparison", "interactive", "all"],
                       default="all", help="Which example to run")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct",
                       help="HuggingFace model name to use for analysis")
    parser.add_argument("--open-browser", action="store_true",
                       help="Automatically open results in browser")
    
    args = parser.parse_args()
    
    print("🧠 LLM Token Probability Analysis System")
    print("="*60)
    print("This system analyzes LLM token generation confidence to identify")
    print("potential areas where the model is uncertain, which may correlate")
    print("with bugs or problematic code generation.")
    print("="*60)
    
    html_files = []
    
    try:
        if args.example in ["simple", "all"]:
            html_file = simple_text_generation_example(model_name=args.model)
            html_files.append(html_file)

        if args.example in ["comparison", "all"]:
            html_file = comparison_example(model_name=args.model)
            html_files.append(html_file)

        if args.example in ["advanced", "all"]:
            html_file = advanced_code_analysis_example(model_name=args.model)
            html_files.append(html_file)

        if args.example == "interactive":
            interactive_demo(model_name=args.model)
        
        # Open browser if requested
        if args.open_browser and html_files:
            print(f"\nOpening {len(html_files)} result(s) in browser...")
            for html_file in html_files:
                webbrowser.open(f"file://{os.path.abspath(html_file)}")
        
        print("\n" + "="*60)
        print("EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        if html_files:
            print("Generated files:")
            for html_file in html_files:
                print(f"  • {html_file}")
        print("\nYou can open these HTML files in your browser to see the visualizations.")
        
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Make sure you have all required dependencies installed:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()