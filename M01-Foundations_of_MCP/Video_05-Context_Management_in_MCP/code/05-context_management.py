import time
import random


# --- Context Management Example ---

# In MCP, "context" refers to the shared information or state
# that is passed between different steps or components of a workflow.
# It allows different parts of the process to access and modify
# data relevant to the overall task.

# A simple dictionary is often used for context.
def initialize_context(initial_data=None):
    """Creates the initial context dictionary."""
    print("[CONTEXT] Initializing context...")
    context = {
        'start_time': time.time(),
        'status': 'Initialized',
        'data': {},
        'errors': [],
        'workflow_id': f"wf_{random.randint(1000, 9999)}"
    }
    if initial_data:
        context.update(initial_data)
    print(f"[CONTEXT] Initial context created: {context}")
    return context


# --- Simulated Workflow Steps ---

def step_1_gather_data(context):
    """Simulates gathering initial data and updating the context."""
    print("\n--- Step 1: Gathering Data ---")
    try:
        # Simulate fetching data from a source
        print("[STEP 1] Fetching user data...")
        time.sleep(0.5)  # Simulate network delay
        user_id = random.randint(1, 100)
        user_data = {'user_id': user_id, 'name': f"User_{user_id}", 'email': f"user{user_id}@example.com"}

        # Update the context
        context['data']['user_info'] = user_data
        context['status'] = 'Data Gathered'
        print(f"[STEP 1] User data gathered for ID: {user_id}")
        print(f"[CONTEXT] Context updated: Status='{context['status']}', Data added.")

    except Exception as e:
        print(f"[STEP 1 ERROR] Failed to gather data: {e}")
        context['errors'].append(f"Step 1 Error: {e}")
        context['status'] = 'Error in Step 1'

    return context  # Return the modified context


def step_2_process_data(context):
    """Simulates processing the data stored in the context."""
    print("\n--- Step 2: Processing Data ---")
    if context['status'].startswith('Error'):
        print("[STEP 2] Skipping due to previous error.")
        return context

    try:
        print("[STEP 2] Processing user data from context...")
        if 'user_info' in context.get('data', {}):
            user_data = context['data']['user_info']
            # Simulate some processing - e.g., validating email
            is_valid_email = '@' in user_data.get('email', '')
            print(f"[STEP 2] Email validation result: {is_valid_email}")

            # Update context with processing results
            context['data']['processing_results'] = {
                'email_valid': is_valid_email,
                'processed_at': time.time()
            }
            context['status'] = 'Data Processed'
            print("[CONTEXT] Context updated: Status='Data Processed', Processing results added.")
        else:
            raise ValueError("User info not found in context data.")

    except Exception as e:
        print(f"[STEP 2 ERROR] Failed to process data: {e}")
        context['errors'].append(f"Step 2 Error: {e}")
        context['status'] = 'Error in Step 2'

    return context


def step_3_generate_report(context):
    """Simulates generating a report using data from the context."""
    print("\n--- Step 3: Generating Report ---")
    if context['status'].startswith('Error'):
        print("[STEP 3] Skipping due to previous error.")
        return context

    try:
        print("[STEP 3] Generating report based on context...")
        user_info = context.get('data', {}).get('user_info', {})
        processing_results = context.get('data', {}).get('processing_results', {})

        if not user_info or not processing_results:
            raise ValueError("Required data for report not found in context.")

        report = f"""
        ----- Workflow Report -----
        Workflow ID: {context.get('workflow_id', 'N/A')}
        User ID:     {user_info.get('user_id', 'N/A')}
        User Name:   {user_info.get('name', 'N/A')}
        Email Valid: {processing_results.get('email_valid', 'N/A')}
        ---------------------------
        """
        print("[STEP 3] Report generated:")
        print(report)

        # Update context
        context['data']['report'] = report
        context['status'] = 'Report Generated'
        context['end_time'] = time.time()
        context['duration'] = context['end_time'] - context['start_time']
        print("[CONTEXT] Context updated: Status='Report Generated', Report added, End time recorded.")

    except Exception as e:
        print(f"[STEP 3 ERROR] Failed to generate report: {e}")
        context['errors'].append(f"Step 3 Error: {e}")
        context['status'] = 'Error in Step 3'

    return context


# --- Main Workflow Execution ---

if __name__ == "__main__":
    print("=== Starting MCP Workflow with Context Management ===")

    # 1. Initialize the context
    workflow_context = initialize_context({'initial_param': 'example_value'})

    # 2. Execute steps, passing and updating the context
    workflow_context = step_1_gather_data(workflow_context)
    workflow_context = step_2_process_data(workflow_context)
    workflow_context = step_3_generate_report(workflow_context)

    # 3. Final Context State
    print("\n=== Workflow Complete ===")
    print(f"Final Status: {workflow_context.get('status', 'Unknown')}")
    if workflow_context.get('errors'):
        print(f"Errors Encountered: {workflow_context['errors']}")
    if 'duration' in workflow_context:
        print(f"Total Duration: {workflow_context['duration']:.4f} seconds")

    # print("\nFinal Context:")
    # import json
    # print(json.dumps(workflow_context, indent=2, default=str)) # Pretty print final context

    # Best Practices Reminder (as comments):
    # - Keep context focused: Only include data relevant to the workflow.
    # - Avoid overly large context: Large objects can impact performance. Pass references or IDs if needed.
    # - Clear Naming: Use descriptive keys within the context dictionary.
    # - Immutability (where possible): If a step doesn't need to modify context, pass a copy or ensure it doesn't mutate.
    # - Error Handling: Use the context to track errors across steps.
