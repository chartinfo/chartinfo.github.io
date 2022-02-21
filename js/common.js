const toggleHidden = (id, shouldShow) => {
    const item = document.getElementById(id);
    if (shouldShow !== undefined) {
      if (shouldShow) {
        item.classList.remove("hidden");
      } else {
        item.classList.add("hidden");
      }
    } else {
      if (item.classList.contains("hidden")) {
        item.classList.remove("hidden");
      } else {
        item.classList.add("hidden");
      }
    }
  };