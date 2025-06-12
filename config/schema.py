# config/schema.py

CLIENT_COLUMNS = {
    "name": "Client Name",
    "status": "status",     
    "email": "Email",
    "id": "client_id"
}

ATTORNEY_COLUMNS = {
    "name": "name",
    "status": "employee_status",
    "email": "attorney_email",
    "id": "attorney_id"
}

MATTER_COLUMNS = {
    "title": "name",
    "status": "status",
    "client_id": "client_id",
    "attorney_id": "attorney_id",
    "created": "open_date"
}

LEAVE_COLUMNS = {
    "attorney_id": "attorney_id",
    "start": "open_date",
    "end": "actual_close_date",
    "status": "status"
}
