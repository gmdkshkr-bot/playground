 /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:669 in code_to_exec                                     

                                                                                

  /mount/src/playground/app2.py:380 in <module>                                 

                                                                                

    377 │   if st.button("🧹 Reset Record", help="Clears all accumulated recei  

    378 │   │   st.session_state.all_receipts_items = []                        

    379 │   │   st.session_state.all_receipts_summary = []                      

  ❱ 380 │   │   st.experimental_rerun()                                         

    381                                                                         

────────────────────────────────────────────────────────────────────────────────

AttributeError: module 'streamlit' has no attribute 'experimental_rerun'
