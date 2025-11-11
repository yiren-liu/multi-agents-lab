"""
Simplified AutoGen Demo - Interview Platform Product Planning

This is a lightweight version for quick testing and understanding the workflow.
It demonstrates multi-agent collaboration by having each agent generate responses.
"""

from datetime import datetime
from config import Config, WorkflowConfig
import json

# Try to import OpenAI client
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: OpenAI client is not installed!")
    print("Please run: pip install -r ../requirements.txt")
    exit(1)


class SimpleInterviewPlatformWorkflow:
    """Simplified workflow for interview platform planning"""

    def __init__(self):
        """Initialize the workflow"""
        if not Config.validate_setup():
            print("ERROR: Configuration validation failed!")
            exit(1)

        self.client = OpenAI(api_key=Config.OPENAI_API_KEY, base_url=Config.OPENAI_API_BASE)
        self.outputs = {}
        self.model = Config.OPENAI_MODEL

    def run(self):
        """Execute the complete workflow"""
        print("\n" + "="*80)
        print("AUTOGEN INTERVIEW PLATFORM WORKFLOW - SIMPLIFIED DEMO")
        print("="*80)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: {self.model}\n")

        # Phase 1: Research
        self.phase_research()

        # Phase 2: Analysis
        self.phase_analysis()

        # Phase 3: Blueprint
        self.phase_blueprint()

        # Phase 4: Review
        self.phase_review()

        # Summary
        self.print_summary()

    def phase_research(self):
        """Phase 1: Market Research"""
        print("\n" + "="*80)
        print("PHASE 1: MARKET RESEARCH")
        print("="*80)
        print("[ResearchAgent is analyzing the market...]")

        system_prompt = """You are a market research analyst. Provide a brief analysis of
3 competitors in AI interview platforms (HireVue, Pymetrics, Codility).
List their key features and identify market gaps in 150 words."""

        user_message = "Analyze the current market for AI-powered interview platforms."

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=Config.AGENT_TEMPERATURE,
            max_tokens=Config.AGENT_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        self.outputs["research"] = response.choices[0].message.content
        print("\n[ResearchAgent Output]")
        print(self.outputs["research"][:500] + "..." if len(self.outputs["research"]) > 500 else self.outputs["research"])

    def phase_analysis(self):
        """Phase 2: Opportunity Analysis"""
        print("\n" + "="*80)
        print("PHASE 2: OPPORTUNITY ANALYSIS")
        print("="*80)
        print("[AnalysisAgent is identifying opportunities...]")

        system_prompt = """You are a product analyst. Based on the market research provided,
identify 3 key market opportunities or gaps for a new AI interview platform.
Be concise in 150 words."""

        user_message = f"""Market research findings:
{self.outputs['research']}

Now identify market opportunities and gaps."""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=Config.AGENT_TEMPERATURE,
            max_tokens=Config.AGENT_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        self.outputs["analysis"] = response.choices[0].message.content
        print("\n[AnalysisAgent Output]")
        print(self.outputs["analysis"][:500] + "..." if len(self.outputs["analysis"]) > 500 else self.outputs["analysis"])

    def phase_blueprint(self):
        """Phase 3: Product Blueprint"""
        print("\n" + "="*80)
        print("PHASE 3: PRODUCT BLUEPRINT")
        print("="*80)
        print("[BlueprintAgent is designing the product...]")

        system_prompt = """You are a product designer. Based on the market analysis and opportunities,
create a brief product blueprint including:
- Key features (3-5)
- User journey (2-3 steps)
Keep it concise - 150 words."""

        user_message = f"""Market Analysis:
{self.outputs['analysis']}

Create a product blueprint for our platform."""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=Config.AGENT_TEMPERATURE,
            max_tokens=Config.AGENT_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        self.outputs["blueprint"] = response.choices[0].message.content
        print("\n[BlueprintAgent Output]")
        print(self.outputs["blueprint"][:500] + "..." if len(self.outputs["blueprint"]) > 500 else self.outputs["blueprint"])

    def phase_review(self):
        """Phase 4: Strategic Review"""
        print("\n" + "="*80)
        print("PHASE 4: STRATEGIC REVIEW")
        print("="*80)
        print("[ReviewerAgent is providing recommendations...]")

        system_prompt = """You are a product reviewer and strategist. Review the product blueprint
and provide 3 strategic recommendations for success.
Be concise - 150 words."""

        user_message = f"""Product Blueprint:
{self.outputs['blueprint']}

Provide strategic review and recommendations."""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=Config.AGENT_TEMPERATURE,
            max_tokens=Config.AGENT_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        self.outputs["review"] = response.choices[0].message.content
        print("\n[ReviewerAgent Output]")
        print(self.outputs["review"][:500] + "..." if len(self.outputs["review"]) > 500 else self.outputs["review"])

    def print_summary(self):
        """Print final summary"""
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)

        print("""
This workflow demonstrated a 4-agent collaboration:
1. ResearchAgent - Analyzed the market
2. AnalysisAgent - Identified opportunities
3. BlueprintAgent - Designed the product
4. ReviewerAgent - Provided strategic recommendations

Each agent received context from the previous agent's output,
demonstrating the sequential workflow pattern of AutoGen.
""")

        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)


if __name__ == "__main__":
    try:
        workflow = SimpleInterviewPlatformWorkflow()
        workflow.run()
        print("\n✅ Workflow completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during workflow execution: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Verify OPENAI_API_KEY is set in .env")
        print("2. Check your API key has sufficient credits")
        print("3. Verify internet connection")
        print("4. Check OpenAI API status at https://status.openai.com")
        import traceback
        traceback.print_exc()
