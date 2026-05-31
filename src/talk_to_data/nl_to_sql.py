import os
from groq import Groq
from src.utils.config import GROQ_API_KEY
from src.talk_to_data.prompt_templates import SYSTEM_PROMPT, get_sql_prompt
from src.talk_to_data.query_runner import run_query

client = Groq(api_key=GROQ_API_KEY)

# Conversation memory - stores last 5 exchanges
conversation_history = []

def generate_sql(question):
    """Convert natural language to SQL with conversation memory"""
    try:
        # Build messages with history for context
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add last 3 exchanges for memory
        for exchange in conversation_history[-3:]:
            messages.append({"role": "user", "content": exchange["question"]})
            messages.append({"role": "assistant", "content": exchange["sql"]})
        
        # Add current question
        messages.append({"role": "user", "content": get_sql_prompt(question)})
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1,
            max_tokens=300
        )
        sql = response.choices[0].message.content.strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        # Validate SQL to prevent hallucinations
        from src.talk_to_data.prompt_templates import validate_sql
        is_valid, message = validate_sql(sql)
        if not is_valid:
            return None, f"Invalid SQL generated: {message}"
        
        return sql, None
    except Exception as e:
        return None, str(e)

def ask(question):
    """Full pipeline: question -> SQL -> results"""
    global conversation_history
    
    print(f"\n Question: {question}")
    
    # Generate SQL
    sql, err = generate_sql(question)
    if err:
        return {"error": f"SQL generation failed: {err}"}
    print(f" Generated SQL: {sql}")
    
    # Run SQL
    df, err = run_query(sql)
    if err:
        return {"error": f"Query failed: {err}", "sql": sql}
    
    print(f" Results ({len(df)} rows):")
    print(df.to_string(index=False))
    
    # Save to conversation memory
    conversation_history.append({
        "question": question,
        "sql": sql
    })
    # Keep only last 5
    if len(conversation_history) > 5:
        conversation_history.pop(0)
    
    return {
        "question": question,
        "sql": sql,
        "results": df,
        "row_count": len(df)
    }

def clear_memory():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    print(" Conversation memory cleared")

def test_queries():
    """Test 5 working query patterns"""
    questions = [
        "How many people defaulted on their loans?",
        "What is the average income for male and female applicants?",
        "Show default rate by education type",
        "What are the top 5 occupation types with highest default rate?",
        "How many applicants own a car vs don't own a car?"
    ]
    print("="*60)
    print("TESTING 5 QUERY PATTERNS")
    print("="*60)
    for i, q in enumerate(questions, 1):
        print(f"\n--- Query {i} ---")
        result = ask(q)
        if "error" in result:
            print(f" Error: {result['error']}")
        print("-"*40)