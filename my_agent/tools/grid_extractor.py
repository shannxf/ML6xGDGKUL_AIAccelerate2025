# my_agent/tools/grid_extractor.py
def grid_extractor(grid: str) -> str:
    """
    Input: 5x7 grid as multiline string
    Output: sentence reading left→right, top→bottom
    """
    lines = [line.strip() for line in grid.split('\n') if line.strip()]
    if len(lines) != 7 or any(len(line) != 5 for line in lines):
        return "Invalid grid"
    
    sentence = ""
    for col in range(5):
        for row in range(7):
            sentence += lines[row][col]
        sentence += " "
    return sentence.strip().capitalize() + "."