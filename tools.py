def search_long_text(table,text_column,query,filters,top_k=10):
    print("searching paragraph...")
    """
    ใช้หา text ยาว เช่น:
    - note
    - description
    - complaint
    - document
    - ticket content
    """
    retrieved_chunks = "67 คน คือคำตอบ"
    return retrieved_chunks

def query_single_table(table,select,where,group_by,order_by,limit=100):
    """
    ใช้ execute SQL จริง
    ต้อง validate ก่อนเสมอ
    """
    result = "67 คือคำตอบ"
    return result

def query_join_tables(tables,join_path,select,where,group_by,order_by,limit=100) :
    result = "67 คือคำตอบ"
    print("searching multiple table...")
    return result


def get_sales_context () :
    print("using sales_context")
    return "test"

def get_customer_cs_context () :
    print("using customer_cs_context")
    return "test"

def get_policy_context () :
    print("using policy_context")
    return "test"

def get_vendor_shipping_context () :
    print("using vendor_shipping_context")
    return "test"

def get_finance_bank_context () :
    print("using finance_bank_context")
    return "test"

def get_inventory_context () :
    print("using inventory_context")
    return "test"

def get_employee_context () :
    print("using employee_context")
    return "test"

def get_document_render_context () :
    print("using document_render_context")
    return "test"

def get_report_context () :
    print("using report_context")
    return "test"