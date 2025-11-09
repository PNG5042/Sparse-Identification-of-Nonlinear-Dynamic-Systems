@echo off
REM run.bat - deterministic demo: create DB and insert a user
python -c "from Project.app import create_and_insert_demo; print(create_and_insert_demo())"
echo EXPECTED: inserted: demo-user