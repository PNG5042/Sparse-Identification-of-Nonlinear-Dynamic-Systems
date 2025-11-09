#!/bin/bash

# ============================================================================
# Materials Analysis Environment Setup and Execution Script
# ============================================================================

set -e  # Exit on error

echo "========================================================================"
echo "Materials Analysis - Setup and Execution Script"
echo "========================================================================"
echo ""

# Define virtual environment name
VENV_NAME="materials_env"

# Check if virtual environment exists
if [ ! -d "$VENV_NAME" ]; then
    echo "📦 Virtual environment not found. Creating new environment..."
    python3 -m venv $VENV_NAME
    echo "✅ Virtual environment created: $VENV_NAME"
    echo ""
else
    echo "✅ Virtual environment found: $VENV_NAME"
    echo ""
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source $VENV_NAME/bin/activate
echo "✅ Environment activated"
echo ""

# Check if packages are installed
echo "📋 Checking required packages..."
PACKAGES_NEEDED=false

for package in numpy pandas scipy matplotlib pysindy scikit-learn; do
    if ! python -c "import ${package//-/_}" 2>/dev/null; then
        PACKAGES_NEEDED=true
        echo "   ❌ $package not found"
    else
        echo "   ✅ $package installed"
    fi
done

echo ""

# Install packages if needed
if [ "$PACKAGES_NEEDED" = true ]; then
    echo "📥 Installing required packages..."
    pip install --upgrade pip
    pip install numpy pandas scipy matplotlib pysindy scikit-learn
    echo "✅ All packages installed"
    echo ""
else
    echo "✅ All required packages already installed"
    echo ""
fi

# Display menu
echo "========================================================================"
echo "Select script to run:"
echo "========================================================================"
echo "1) Run tensile_test.py"
echo "2) Run creep_test.py"
echo "3) Run both scripts"
echo "4) Exit"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "========================================================================"
        echo "Running tensile_test.py"
        echo "========================================================================"
        echo ""
        if [ -f "tensile_test.py" ]; then
            python tensile_test.py
        else
            echo "❌ Error: tensile_test.py not found in current directory"
            exit 1
        fi
        ;;
    2)
        echo ""
        echo "========================================================================"
        echo "Running creep_test.py"
        echo "========================================================================"
        echo ""
        if [ -f "creep_test.py" ]; then
            python creep_test.py
        else
            echo "❌ Error: creep_test.py not found in current directory"
            exit 1
        fi
        ;;
    3)
        echo ""
        echo "========================================================================"
        echo "Running tensile_test.py"
        echo "========================================================================"
        echo ""
        if [ -f "tensile_test.py" ]; then
            python tensile_test.py
        else
            echo "❌ Error: tensile_test.py not found in current directory"
            exit 1
        fi
        
        echo ""
        echo "========================================================================"
        echo "Running creep_test.py"
        echo "========================================================================"
        echo ""
        if [ -f "creep_test.py" ]; then
            python creep_test.py
        else
            echo "❌ Error: creep_test.py not found in current directory"
            exit 1
        fi
        ;;
    4)
        echo ""
        echo "👋 Exiting..."
        deactivate
        exit 0
        ;;
    *)
        echo ""
        echo "❌ Invalid choice. Exiting..."
        deactivate
        exit 1
        ;;
esac

echo ""
echo "========================================================================"
echo "✅ Execution complete!"
echo "========================================================================"
echo ""
echo "Generated files:"
ls -lh *.csv 2>/dev/null || echo "No CSV files generated"
echo ""
echo "To deactivate virtual environment, run: deactivate"
echo ""
