from pydantic import Field

from OpenManus.app.agent.toolcall import ToolCallAgent
from OpenManus.app.config import config
from OpenManus.app.prompt.visualization import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from OpenManus.app.tool import Terminate, ToolCollection
from OpenManus.app.tool.chart_visualization.chart_prepare import VisualizationPrepare
from OpenManus.app.tool.chart_visualization.data_visualization import DataVisualization
from OpenManus.app.tool.chart_visualization.python_execute import NormalPythonExecute


class DataAnalysis(ToolCallAgent):
    """
    A data analysis agent that uses planning to solve various data analysis tasks.

    This agent extends ToolCallAgent with a comprehensive set of tools and capabilities,
    including Data Analysis, Chart Visualization, Data Report.
    """

    name: str = "DataAnalysis"
    description: str = "An analytical agent that utilizes multiple tools to solve diverse data analysis tasks"

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 15000
    max_steps: int = 20

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            NormalPythonExecute(),
            VisualizationPrepare(),
            DataVisualization(),
            Terminate(),
        )
    )
