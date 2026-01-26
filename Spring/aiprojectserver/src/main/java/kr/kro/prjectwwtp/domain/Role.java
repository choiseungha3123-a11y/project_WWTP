package kr.kro.prjectwwtp.domain;

import com.fasterxml.jackson.annotation.JsonProperty;

public enum Role {
	@JsonProperty("ROLE_MEMBER")
	ROLE_MEMBER,
	@JsonProperty("ROLE_ADMIN")
	ROLE_ADMIN;
}

