import streamlit as st
from openai import OpenAI
import psycopg2
import re
import os
import pandas as pd

# Set up OpenAI API key
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Database connection parameters
db_params = st.secrets["db_credentials"]

def connect_to_db():
    """Establish a connection to the Redshift database."""
    try:
        conn = psycopg2.connect(**db_params)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to the database: {e}")
        return None

def execute_query(conn, query):
    """Execute a SQL query and return the results and cursor."""
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
            return results, cur
    except psycopg2.Error as e:
        print(f"Error executing query: {e}")
        return None, None

def get_gpt4_response(prompt):
    """Get a response from GPT-4 using the OpenAI API."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates SQL queries for a Redshift database. The database contains information about mentees, courses, lessons, companies, and more. Always return only the SQL query, without any explanations, comments, or formatting."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return None

def clean_sql_query(sql_query):
    """Remove any Markdown formatting or unnecessary characters from the SQL query."""
    sql_query = re.sub(r'```\w*\n?', '', sql_query)
    sql_query = sql_query.strip()
    return sql_query

def generate_sql_query(user_input):
    """Generate a SQL query based on the user's input."""
    reference_logic = """
    Logic from reference query:
    1. Use a CTE (Common Table Expression) named 'cte' for complex calculations.
    2. Join these main tables: mentee_lessons, batch_lessons, lessons, batches, super_batches, super_batch_groups, academy_topics, academy_modules, super_batch_academy_topics, courses, mentees, mentee_modules, mentee_batches, users, interviewbit_tests, interviewbit_test_problems, problems, interviewbit_test_sessions, interviewbit_test_session_problems.
    3. Calculate program using CASE statement based on course slug (case when c.slug in ('scaler-wp-intake','scaler-wp-elitex','scaler-wp-elitex-slow','scaler-wp-superx','scaler-wp-beginner','scaler-wp-beginner-refresher',
        'scaler-wp-beginner-repeater-failed-once','scaler-wp-beginner-repeater-failed-twice','scaler-wp-data-engineering','scaler-wp-elitex-refresher-java','scaler-wp-elitex-refresher-python',
        'scaler-wp-fullstack','scaler-wp-backend')
        then 'Academy'
    when c.slug in ('scaler-ds-intake','scaler-ds-elitex','scaler-ds-elitex-refresher','scaler-ds-superx','scaler-ds-beginner','scaler-ds-beginner-refresher',
        'scaler-ds-analytics','scaler-ds-beginner-without-python')
        then 'DSML'
    when c.slug in ('scaler-elitex', 'scaler-superx', 'scaler-plus', 'techversity-beginner', 'techversity-advanced')
        then 'Old Academy'
    when c.slug in ('scaler-wp-us-intake','scaler-wp-us-elitex','scaler-wp-us-superx','scaler-wp-us-beginner', 'scaler-crash-course',
        'scaler-ug-primary')
        then 'US Academy'
    else c.slug 
    end) as program
    4. Always convert datetime columns to date type
    6. Calculate class_type using CASE statement based on lesson_type and lecture_bucket_id.
    7. Calculate live_attendance, recorded_attendance, and overall_attendance using LEAST and MAX functions.
    8. Count assignment and homework problems (total and solved) based on test_type.
    9. Calculate assignment_psp and hw_psp as percentages of solved problems.
    10. Exclude Scaler and InterviewBit emails.
    11. Group by user_id, email, batch name, super_batch_name, program, module_name, order, class_topic, class_date, class_timing, and lecture_bucket_id.
    12. Order by email, module order, and class_date.
    13. Do not include any default filters or exclusions unless specified in the user's request.
    14. Whenever the batch name is needed use the name from the super_batches table.
    15. whenever class date is needed use super batch academy topics table and column name date_of_topic. And remember to join super batch academy topics table use academy topic id and super batch id 
    16. Aggregates are not allowed in WHERE clause
    17. TYPECAST THE DATETIME COLUMNS TO DATE
    18. always remember when any of these are used apply this formula in case of learner level data
    ,LEAST(MAX(CASE 
				WHEN l.lesson_type in (0,6)
					THEN COALESCE(ml.attendance, 0)
				END) / LEAST(MAX(bl.adjustment_factor), 0.8), 100) AS live_attendance
	,LEAST(MAX(CASE 
				WHEN l.lesson_type in (0,6)
					THEN COALESCE(ml.passive_attendance, 0)
				END) / LEAST(MAX(bl.adjustment_factor), 0.8), 100) AS recorded_attendance
    ,LEAST(MAX(CASE 
				WHEN l.lesson_type in (0,6)
					THEN COALESCE(ml.aggregate_attendance, 0)
				END) / LEAST(MAX(bl.adjustment_factor), 0.8), 100) AS overall_attendance
	,count(DISTINCT CASE 
			WHEN ibt.test_type = 2
				THEN ibtp.id
			END) AS total_assignment_problems
	,count(DISTINCT CASE 
			WHEN ibtsp.STATUS = 2
				AND ibt.test_type = 2
				THEN ibtsp.id
			END) AS total_assignment_problems_solved
	,CASE 
		WHEN total_assignment_problems != 0
			THEN ((total_assignment_problems_solved * 1.00) / total_assignment_problems) * 100
		END AS assignment_psp
	,count(DISTINCT CASE 
			WHEN ibt.test_type = 3
				THEN ibtp.id
			END) AS total_hw_problems
	,count(DISTINCT CASE 
			WHEN ibtsp.STATUS = 2
				AND ibt.test_type = 3
				THEN ibtsp.id
			END) AS total_hw_problems_solved
	,CASE 
		WHEN total_hw_problems != 0
			THEN ((total_hw_problems_solved * 1.00) / total_hw_problems) * 100
		END AS hw_psp
    19. REMEMBER WHENEVER THERE IS SOME AGGREGATE DATA ASKED WHETHER BE ON CLASS LEVEL OR LEARNER LEVEL OR BATCH LEVEL USE LEARNER LEVEL DATA AS BASE DATA.
    21. class name is always as academy topics table with column name title, module name is always academy modules table with column name name
    22. ALWAYS CONVERT COLUMN WITH DATA TYPE AS DATETIME TO DATE DATA TYPE USING ::DATE
    23. REMEMBER TO KEEP THE SYNTAX AS REDSHIFT QUERY
    24. HERE IS DETAIL OF HOW TO CONNECT THE TABLES 
    TABLE MENTEE LESSONS IS CONNECTED WITH MENTEES USING MENTEE ID 
    TABLE BATCH LESSONS IS CONNECTED WITH MENTEE LESSONS USING BATCH LESSON ID
    TABLE LESSONS IS CONNECTED WITH BATCH LESSONS USING LESSON ID
    TABLE ACADEMY TOPICS IS CONNECTED WITH LESSONS USING ACADEMY TOPIC ID
    TABLE BATCHES IS CONNECTED WITH BATCH LESSONS USING BATCH ID 
    TABLE SUPER BATCHES USED FOR GETTING BATCH NAME IS CONNECTED WITH BATCHES USING BATCH ID 
    TABLE SUPER BATCH ACADEMY TOPICS IS CONNECTED USING SUPER BATCH ID IN SUPER BATCHES TABLE AND ID IN ACADEMY TOPICS TABLE 
    WHENEVER IN WHERE CLAUSE DATE IS USED THEN DO DATE_COLUMN::DATE= CONDITION
    25. ALWAYS MAKE THE QUERY IN THIS ORDER :
        mentees -> mentee lessons -> batch lessons -> batches -> super batches -> lessons -> academy topics -> super batch academy topics
    26. ALWAYS USE mentee lessons status=1 and batch lessons status=3 whenever these tables are used
    27. FOR ASSIGNMENT PSP OR ANY PROBLEM SOLVING METRIC USE
    LEFT JOIN scaler_ebdb_interviewbit_tests ibt ON ibt.id = l.test_id
    LEFT JOIN scaler_ebdb_interviewbit_test_problems ibtp ON ibtp.test_id = ibt.id
    LEFT JOIN scaler_ebdb_interviewbit_test_sessions ibts ON ibts.test_id = ibt.id
	AND users.id = ibts.user_id
    LEFT JOIN scaler_ebdb_interviewbit_test_session_problems ibtsp ON ibtsp.test_session_id = ibts.id
    28. WHENEVER ASSIGNMENT PSP OR HW PSP IS ASKED CALCULATE THE total_assignment_problems_solved AND total_assignment_problems FIRSTLY AND AFTER THAT GIVE THE PERCENTAGE
    29. In where clause do not use any subquery
    """

    schema_info = """
    Available tables and their relevant columns:
    - scaler_ebdb_mentee_lessons: id, mentee_id, batch_lesson_id, status, attendance, passive_attendance, aggregate_attendance, lesson_rating
    - scaler_ebdb_batch_lessons: id, batch_id, lesson_id, status, adjustment_factor
    - scaler_ebdb_lessons: id, academy_topic_id, lesson_type, test_id
    - scaler_ebdb_batches: id, super_batch_id
    - scaler_ebdb_super_batches: id, name, course_id, super_batch_group_id
    - scaler_ebdb_super_batch_groups: id, group_name
    - scaler_ebdb_academy_topics: id, title, academy_module_id
    - scaler_ebdb_academy_modules: id, name, category
    - scaler_ebdb_super_batch_academy_topics: super_batch_id, academy_topic_id, date_of_topic, lecture_bucket_id
    - scaler_ebdb_courses: id, title, slug
    - scaler_ebdb_mentees: id, user_id, status
    - scaler_ebdb_mentee_modules: id, mentee_id, academy_module_id, status, order
    - scaler_ebdb_mentee_batches: mentee_id, batch_id
    - scaler_ebdb_users: id, email, phone_number
    - scaler_ebdb_interviewbit_tests: id, test_type
    - scaler_ebdb_interviewbit_test_problems: id, test_id, problem_id
    - scaler_ebdb_problems: id
    - scaler_ebdb_interviewbit_test_sessions: id, test_id, user_id
    - scaler_ebdb_interviewbit_test_session_problems: id, test_session_id, problem_id, status
    - scaler_ebdb_mentees: id, status, solved_percentage
    - scaler_ebdb_mentee_batches: mentee_id, batch_id
    - scaler_ebdb_batches: id, batch_name, super_batch_id
    - scaler_ebdb_super_batches: id, name, course_id
    - scaler_ebdb_courses: id, title
    - scaler_ebdb_super_batch_academy_topics: super_batch_id, academy_topic_id, date_of_topic
    - scaler_ebdb_academy_topics: id, title, academy_module_id
    - scaler_ebdb_lessons: id, academy_topic_id, lesson_types
    - scaler_ebdb_batch_lessons: id, batch_id, lesson_id, start_time, end_time, status
    - scaler_ebdb_mentee_lessons: id, mentee_id, batch_lesson_id, status, attendance, passive_attendance, aggregate_attendance, submitted_feedback
    - scaler_ebdb_companies: id, name, slug, use_dashboard
    - scaler_ebdb_company_team_members: id, company_team_id, user_id, member_type, access_levels
    - scaler_ebdb_company_teams: id, name, company_id, allocated, state
    - scaler_ebdb_interviewbit_test_accounts: id, company_id, email, company_name
    - scaler_ebdb_interviewbit_test_problems: id, test_id, problem_id, max_score, custom_question_name, lock, message
    - scaler_ebdb_problem_tests: id, problem_id, score, time_limit, memory_limit, created_at, updated_at, input_data_file_name, output_data_file_name, input_lines_per_test, test_cases_count
    - scaler_ebdb_interviewbit_test_session_codes: id, test_session_id, problem_id, programming_language_id, user_saved_code_id, session_type
    - scaler_ebdb_interviewbit_test_session_problems: id, test_session_id, problem_id, score, status, start_time, time_to_solve, first_solved_at, practice_score, penalty_time
    - scaler_ebdb_interviewbit_test_sessions: id, test_id, user_id, candidate_name, candidate_email, candidate_work_experience, candidate_gender, candidate_contact_number, candidate_city, candidate_cgpa, candidate_degree, candidate_branch, candidate_university, candidate_graduation_city, disclaimer, status, user_start_time, user_url_identifier, score, restricted_events, owner_session, opened_at, clicked_at, slug, platform_feedback, question_feedback, practice_score
    - scaler_ebdb_interviewbit_test_user_submissions: id, test_id, test_session_id, problem_id, user_submission_id, submission_type
    - scaler_ebdb_interviewbit_tests: id, user_id, test_name, test_duration, problem_count, cutoff, total_score, candidate_email, candidate_city, candidate_name, candidate_gender, candidate_branch, candidate_work_experience, candidate_cgpa, candidate_contact_number, candidate_university, candidate_resume, candidate_graduation_city, candidate_degree, company_id, start_time, end_time, instructions, slug, lock
    - scaler_ebdb_problems: id, problem_statement, judge_type
    - scaler_ebdb_user_submissions: id, user_id, problem_id, submission_type, programming_language_id, submission_content, status, result
    - scaler_ebdb_user_submission_test_results: id, user_submission_id, user_id, problem_test_id, result, score, time, memory, first_failed_test_case, failed_input, expected_output, user_output, error_message, created_at, updated_at
    - scaler_ebdb_users: id, email, name, username, slug
    - scaler_ebdb_user_saved_codes: id, user_id, problem_id, programming_language_id, content
    - scaler_ebdb_academy_modules: id, name, category
    - scaler_ebdb_course_modules: id, course_id, academy_module_id
    - scaler_ebdb_super_batch_modules: id, supere_batch_id, academy_module_id
    - scaler_ebdb_mentee_modules: id, mentee_id, academy_module_id
    - scaler_ebdb_mentee_session_grants: id, mentee_id, category, grant_type, owner_type, owner_id
    - scaler_ebdb_ninja_hire_interviews: id, type
    - scaler_ebdb_ninja_hire_associations: ninja_hire_interview_id, associate
    - scaler_ebdb_notebooks: id, name, directory_type, parent_id
    - scaler_ebdb_notebook_owner: id, user_id, notebook_id, access
    - scaler_ebdb_notebook_resources: id, resource_type, resource_id, notebook_id
    - scaler_ebdb_notebook_associations: id, associate_type, associate_id, notebook_id
    - scaler_ebdb_notes: id, content
    """

    prompt = f"""Given the following reference logic and schema information:

Reference Logic:
{reference_logic}

Schema Information:
{schema_info}

Generate a SQL query for the following request: {user_input}

Return only the SQL query, without any explanations, comments, or formatting. Use the DISTINCT keyword where appropriate. Follow the structure and logic of the reference query, adapting it to the specific request."""

    generated_sql = get_gpt4_response(prompt)
    return clean_sql_query(generated_sql)

def main():
    # Set page configuration
    st.set_page_config(page_title="Scaler Academy Database Chatbot", page_icon="🤖", layout="wide")

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
    }
    .stTextInput>div>div>input {
        background-color: white;
    }
    .footer {
        text-align: center;
        padding: 10px 0;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

    # Application header
    st.title("Scaler Academy Database Chatbot")
    
    st.markdown("---")
    st.write("Ask questions about mentees, courses, lessons, companies, and more.")

    # Initialize connection
    if 'conn' not in st.session_state:
        st.session_state.conn = connect_to_db()

    # Check if connection is successful
    if not st.session_state.conn:
        st.error("Failed to connect to the database. Please check your connection settings.")
        return

    # User input
    user_input = st.text_input("Enter your question:", placeholder="e.g., Show me the top 5 mentees by attendance")

    if st.button("Submit", key="submit"):
        if user_input:
            with st.spinner("Generating query and fetching results..."):
                # Print user input
                st.subheader("Your question:")
                st.info(user_input)

                generated_sql = generate_sql_query(user_input)

                if generated_sql:
                    st.subheader("Generated SQL query:")
                    st.code(generated_sql, language="sql")

                    results, cur = execute_query(st.session_state.conn, generated_sql)

                    if results and cur:
                        st.subheader("Query results:")
                        df = pd.DataFrame(results)
                        
                        # Get column names from the cursor description
                        column_names = [desc[0] for desc in cur.description]
                        
                        # Assign column names to the dataframe
                        df.columns = column_names
                        
                        # Display results in an expandable section without highlighting
                        with st.expander("View Results Table", expanded=True):
                            st.dataframe(df, use_container_width=True)

                        # Add download button for CSV
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download results as CSV",
                            data=csv,
                            file_name="query_results.csv",
                            mime="text/csv",
                        )
                    else:
                        st.warning("No results found or there was an error executing the query.")
                else:
                    st.error("I'm sorry, I couldn't generate a proper query for your request.")
        else:
            st.warning("Please enter a question.")

    # Add a centered footer
    st.markdown("---")
    st.markdown('<p class="footer">Built with ❤️ by the Scaler Product Analytics Team</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
