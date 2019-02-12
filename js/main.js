
var current_view = "home";

function update_view(new_view){
	var curr_div = document.getElementById(current_view);
	curr_div.style.display = "None";
	
	current_view = new_view;
	curr_div = document.getElementById(current_view);
	curr_div.style.display = "";
}

