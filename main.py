from orchestrator import orchestrate_tools

def main():
    print("Hello from toolhub!")
    print("Use any of the following tools:")
    print("1. Resume Analyzer.")
    print("2. Email Generator.")
    print("3. Readme Generator.")
    print("4. Explain the codebase.")
    print("5. Code Review.")
    print("6. Code Summarizer.")
    print("7. Web search for a single query.")
    print("8. Job search from S&P 500 companies based on your resume.")
    print("9. Chat with a LLM.")
    print("10. Exit")
    choice = input("Enter your choice: ")

    orchestrate_tools(choice)


if __name__ == "__main__":
    main()
