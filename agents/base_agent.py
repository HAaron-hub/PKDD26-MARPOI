from .agent_tools import DEFAULT_AGENT_TOOLS, get_tool_schemas, invoke_tool


class BaseAgent:
    """
    Base class for all agents with shared interfaces and utilities.
    """
    def __init__(self, mapoi_instance):
        self.mapoi = mapoi_instance
    
    def get_shared_data(self):
        """Get shared data from the MAPOI instance."""
        return {
            'poiInfos': self.mapoi.poiInfos,
            'datasetName': self.mapoi.datasetName,
            'knowledge': self.mapoi.knowledge,
            'transitions': self.mapoi.transitions
        }

    def get_tools(self):
        """Return the tool registry available to agents."""
        return DEFAULT_AGENT_TOOLS

    def get_tool_schemas(self):
        """Return LLM function-calling compatible tool schemas."""
        return get_tool_schemas()

    def call_tool(self, tool_name: str, **kwargs):
        """Invoke a registered agent tool."""
        return invoke_tool(tool_name, **kwargs)
